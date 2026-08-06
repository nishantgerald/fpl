"""Generate the Ed25519 pair the relay is authenticated with.

The private half goes on Heroku as LLM_RELAY_PRIVATE_KEY and never touches this
machine's disk; the public half goes in the relay's environment. Printing them
together, once, is the only moment they share a screen.
"""

import base64

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def main() -> None:
    private = Ed25519PrivateKey.generate()
    private_raw = private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_raw = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    print("On Heroku (the dyny signs with this; it never leaves Heroku):")
    print(f"  LLM_RELAY_PRIVATE_KEY={base64.b64encode(private_raw).decode()}")
    print()
    print("On this machine (the relay only verifies; safe at rest):")
    print(f"  LLM_RELAY_PUBLIC_KEY={base64.b64encode(public_raw).decode()}")


if __name__ == "__main__":
    main()
