# Contributing

## Git Setup

```bash
git clone <REPO_URL>
cd sample-the-physical-ai-toolchain-on-aws
```

> Note: DCV workstation access (port 8443) requires any VPN **disconnected** — VPNs commonly block non-standard ports.

## Development Setup

```bash
cd cdk && npm install        # CDK dependencies
pip install -r training/requirements.txt  # Python dependencies for training scripts
```

## Key Docs

| Doc | What |
|-----|------|
| [TESTING.md](TESTING.md) | Known issues when testing from fresh clone |
| [docs/cosmos-deployment-guide.md](docs/cosmos-deployment-guide.md) | Full Cosmos deployment instructions + troubleshooting |
| [docs/glossary.md](docs/glossary.md) | Physical AI terminology reference |
| [workshop/README.md](workshop/README.md) | Workshop overview + lab sequence |

## AWS Account

- Account: `<YOUR_ACCOUNT_ID>` (derive at runtime; do not hardcode in code — see `tests/test_repo_hygiene.py`)
- Region: `us-west-2` (primary, where the toolchain is validated)
- Secrets in Secrets Manager (regional — keep them in the deploy region): `physical-ai/ngc-api-key`, `physical-ai/nim-api-key`
