from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Final, List

import torch
from freetoken.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg
from freetoken.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)


class SchedulerIOMixin:
    """
    Mixin class for Scheduler I/O operations.

    This class handles the communication between the scheduler and the tokenizer.

    Public Utilities:
        receive_msg: Function to receive messages from the tokenizer.
        send_result: Function to send results back to the tokenizer.
        sync_all_ranks: Function to synchronize all ranks on CPU side.
    """

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        tp_info = config.tp_info
        self.tp_cpu_group: Final = tp_cpu_group
        if config.offline_mode:
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return  # early exit

        if tp_info.is_primary():
            self._recv_from_tokenizer: Final = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,
                decoder=BaseBackendMsg.decoder,
            )
            self._send_into_tokenizer: Final = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=config.backend_create_detokenizer_link,
                encoder=BaseTokenizerMsg.encoder,
            )

        recv = self._recv_msg_single_rank
        send = self._reply_tokenizer_rank0
        if tp_info.size > 1:
            if tp_info.is_primary():
                recv = self._recv_msg_multi_rank0
                self._send_into_ranks: Final = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
                )
            else:
                recv = self._recv_msg_multi_rank1
                send = self._reply_tokenizer_rank1
                self._recv_from_rank0: Final = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr,
                    create=False,
                    decoder=BaseBackendMsg.decoder,
                )

        self.receive_msg = recv
        self.send_result = send

        if tp_info.size > 1:
            self._handshake_rank_relay(tp_info)

    # ------------------------------------------------------------------------------------------
    # The rank-0 -> rank-N request relay is ZeroMQ PUB/SUB. A PUB socket DROPS every message for
    # which no subscription is registered yet, and rank N's connect + SUBSCRIBE travel through
    # ZeroMQ's I/O thread asynchronously -- nothing above waits for them to land. Without a
    # handshake the first request published after readiness can be lost: rank 0 then blocks in
    # the gloo broadcast of `_recv_msg_multi_rank0` waiting for a rank N that blocks in the SUB
    # receive of `_recv_msg_multi_rank1`, no timeout, no error, while the frontend keeps
    # answering GETs (observed: a first request wedged at ready+1 s and was only served at ready+60 s).
    #
    # The handshake: rank 0 publishes a hello frame and every rank reports, through the gloo
    # group (the same `broadcast` primitive the relay already uses), whether it has received one;
    # rank 0 repeats until every subscriber has. Then rank 0 publishes ONE done frame and each
    # subscriber drains hello frames until it sees it -- ZeroMQ preserves order on a connection
    # whose subscription is registered, so the done frame is the last handshake frame that can
    # arrive, and nothing of the handshake can be mistaken for a request later. Only after that
    # does the caller reach `sync_all_ranks()` and the "Scheduler is ready" ack.
    # ------------------------------------------------------------------------------------------
    _RELAY_HELLO: Final = b"\x00freetoken-relay-hello"
    _RELAY_DONE: Final = b"\x00freetoken-relay-done"
    _RELAY_POLL_MS: Final = 50
    _RELAY_MAX_HELLOS: Final = int(os.environ.get("FREETOKEN_RELAY_HANDSHAKE_MAX_HELLOS", "2400"))

    def _handshake_rank_relay(self, tp_info) -> None:
        t0 = time.monotonic()
        size = tp_info.size
        if tp_info.is_primary():
            pub = self._send_into_ranks.socket
            hellos = 0
            while True:
                hellos += 1
                pub.send(self._RELAY_HELLO)
                if self._relay_subscribers_seen(tp_info, seen=False) == size - 1:
                    break
                if hellos >= self._RELAY_MAX_HELLOS:
                    raise RuntimeError(
                        f"TP relay handshake: no subscriber acknowledged after {hellos} hellos "
                        f"({time.monotonic() - t0:.1f} s); the rank-0 PUB never reached rank>=1's SUB"
                    )
            pub.send(self._RELAY_DONE)
            logger.info(
                f"TP relay handshake: {size - 1} subscriber(s) joined after {hellos} hello(s) "
                f"in {(time.monotonic() - t0) * 1000:.0f} ms"
            )
        else:
            sub = self._recv_from_rank0.socket
            seen = False
            while True:
                if not seen and sub.poll(timeout=self._RELAY_POLL_MS):
                    frame = sub.recv()
                    assert frame == self._RELAY_HELLO, f"unexpected relay frame during handshake: {frame!r}"
                    seen = True
                if self._relay_subscribers_seen(tp_info, seen=seen) == size - 1:
                    break
            while True:
                frame = sub.recv()
                if frame == self._RELAY_DONE:
                    break
                assert frame == self._RELAY_HELLO, f"unexpected relay frame during handshake: {frame!r}"

    def _relay_subscribers_seen(self, tp_info, seen: bool) -> int:
        """One round of the handshake: every rank >= 1 broadcasts whether it has received a hello."""
        total = 0
        for root in range(1, tp_info.size):
            flag = torch.tensor(int(seen) if tp_info.rank == root else 0)
            self.tp_cpu_group.broadcast(flag, root=root).wait()
            total += int(flag.item())
        return total

    def run_when_idle(self):
        raise NotImplementedError("should be implemented")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: List[BaseTokenizerMsg]) -> None:
        raise NotImplementedError("should be implemented")

    def sync_all_ranks(self) -> None:
        self.tp_cpu_group.barrier().wait()

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_tokenizer.get())
        while not self._recv_from_tokenizer.empty():
            pending_msgs.append(self._recv_from_tokenizer.get())
        return pending_msgs

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            raw = self._recv_from_tokenizer.get_raw()
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))

        pending_raw_msgs: List[bytes] = []
        while not self._recv_from_tokenizer.empty():
            pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # broadcast the number of raw messages to all ranks
        src_tensor = torch.tensor(len(pending_raw_msgs))
        self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

        for raw in pending_raw_msgs:
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))
        return pending_msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_rank0.get())

        # ensure all ranks have the same number of raw messages
        dst_tensor = torch.tensor(-1)
        self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()
        dst_length = int(dst_tensor.item())

        for _ in range(dst_length):
            pending_msgs.append(self._recv_from_rank0.get())
        return pending_msgs

    def _reply_tokenizer_rank0(self, reply: List[BaseTokenizerMsg]) -> None:
        num_reply = len(reply)
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        if num_reply == 1:
            self._send_into_tokenizer.put(reply[0])
        elif num_reply > 1:
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))  # type: ignore

    def _reply_tokenizer_rank1(self, reply: List[BaseTokenizerMsg]) -> None:
        _ = reply  # do nothing for non-primary ranks
