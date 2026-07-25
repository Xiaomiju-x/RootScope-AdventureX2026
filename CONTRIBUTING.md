# Contributing

Contributions to the public, device-free reference layer are welcome.

Before opening a pull request:

1. Keep all examples synthetic.
2. Do not add model weights, datasets, prompts, calibration data or device logs.
3. Do not add serial, GPIO, relay, pump, motor or network-control code.
4. Do not add credentials, IP addresses, device IDs or private evidence.
5. Preserve the rule that LLM text cannot change an action proposal.
6. Preserve fail-closed behavior for missing, stale, conflicting or OOD evidence.
7. Run `pytest`.

Pull requests that attempt to reconstruct the private hardware control path will
not be accepted in this public repository.

