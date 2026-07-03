"""Public PureChain network configuration for the real on-chain deployment.

Only public information lives here (RPC endpoint, chain ID, wallet address).
Never put a private key in this file or anywhere in the repo -- see
README.md in this folder for how signing keys are handled.
"""

RPC_URL = "https://purechainnode.com:8547"
CHAIN_ID = 900520900520
CURRENCY = "PCC"

# Public wallet address provided by the project owner (read-only reference;
# no private key is stored here or anywhere in this repo).
OWNER_ADDRESS = "0x6aDb63B9dBaa1e6df2bA2D16bA6273aE916D429A"
