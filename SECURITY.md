# Security

Report security issues privately with the "Report a vulnerability" button in this repository's Security tab, not as
public issues. Issues in upstream FreeToken code that also affect upstream should be reported to
[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken/security/advisories/new) as well.

Note: the server has no authentication. Bind it to `127.0.0.1` (the default of `rdna3/serve.sh`) or put an
authenticating reverse proxy in front before exposing it to a network.
