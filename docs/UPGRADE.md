# Upgrading slancha-mesh — security defaults & version skew

New security controls ship **default-OFF** so a binary upgrade never breaks a
running fleet (a node on the old build, or one that hasn't been re-keyed, keeps
working). The flip side: **upgrading the binary does NOT turn them on** — you must
enable them deliberately, fleet-wide, in the right order. This is the migration
sequence + the skew rules.

## Default-OFF security flags

| Flag | Default | What it does when ON | Turn on AFTER |
|------|---------|----------------------|---------------|
| `SLANCHA_REQUIRE_NODE_IDENTITY` | OFF | Registry rejects a heartbeat without a valid Ed25519 `identity_cert` (#102) | every node ships a cert (keygen + `SLANCHA_NODE_KEY_B64`) |
| `SLANCHA_VERIFY_PEERS` | OFF | Discovery drops a federated peer that can't answer a signed challenge (#108) | every peer registry serves a peer key (`SLANCHA_PEER_KEY_B64`) |
| `SLANCHA_NODE_URL_BLOCK_LOOPBACK` / `_BLOCK_PRIVATE` | OFF | Tighten the node_url SSRF guard for multi-tenant cloud | your topology has no legit loopback/private node URLs |

All require the `signing` extra (`pip install 'slancha-mesh[signing]'`) on both ends.

## Migration sequence — turning ON node identity (`#102`)

Enabling `SLANCHA_REQUIRE_NODE_IDENTITY` before every node presents a cert will
**403 the whole fleet**. Order matters:

1. **Upgrade the registry + all nodes** to a build with identity support (this one).
2. **Key every node**: `slancha-mesh node keygen` → persist the secret 0600 (e.g.
   `/etc/slancha/node.key`), set `SLANCHA_NODE_KEY_B64` in the node's environment.
   Nodes now SEND `identity_cert` in every heartbeat; the registry PINS
   `node_id ↔ public_key` on first sight (trust-on-first-use) but does not yet
   require it.
3. **Soak** until the registry has pinned every active node (check the registry
   snapshot — each node shows a pinned key). A node that re-keys here must be
   re-pinned by an operator (a pin change is refused — that's the impersonation
   defense working).
4. **Flip** `SLANCHA_REQUIRE_NODE_IDENTITY=1` on the registry. Now an un-certed or
   wrong-key heartbeat is rejected; a token holder can no longer impersonate a peer.

Roll back by unsetting the flag — the pins persist, so re-enabling is instant.

## Migration sequence — turning ON peer verification (`#108`)

1. Give each peer registry a keypair; set `SLANCHA_PEER_KEY_B64` +
   `SLANCHA_PEER_NODE_ID`. Its `/models` now returns a signed `challenge_response`.
2. Soak until every federated peer serves a key (an un-keyed peer is dropped once
   verification is on).
3. Flip `SLANCHA_VERIFY_PEERS=1` on the asking side. TOFU-pins host→pubkey; drops
   MITM / key-change.

## Version-skew rules

- The registry and nodes speak a forward-compatible heartbeat: an **older node**
  against a **newer registry** works (unknown fields ignored; identity optional
  until required). A **newer node** sending `identity_cert` to an **older
  registry** also works (the field is ignored there).
- Never enable a `REQUIRE_*` flag while any node still runs a build without the
  corresponding sender — that is the one combination that hard-fails.
- The promotion gate / eval-row shapes are a cross-repo contract (see
  `cross-repo-schema-pin.yml`); they are validated in CI, not skewed at runtime.

## Auth model — two planes (don't conflate them) (#127)

mesh has **two** auth surfaces, and the OIDC `Authenticator` seam guards only one:

- **Node/data plane** (`/heartbeat`, `/registry`, `/allocate`, `/gpu/*`): a single
  static bearer (`SLANCHA_NODE_TOKEN`). This is intentional — nodes are fleet peers
  on a trusted network, not per-human identities. Node *identity* (who a node is) is
  the Ed25519 cert above (#102), not OIDC. `require_role` does NOT gate these routes.
- **Operator plane**: routes mounted through the `mesh.auth.get_authenticator` seam,
  where a downstream (e.g. Kanpai's `OidcAuthenticator` + `require_role`) enforces
  per-actor RBAC. OSS mesh ships no operator routes, so `require_role` has no call
  sites here by design — it's the extension point a distribution wires its operator
  API onto.

Practical consequence: overriding `get_authenticator` with OIDC does **not** put
OIDC in front of the data plane. Restrict the data port to the trusted mesh network
(loopback / tailnet / firewall) and rely on the node token + node-identity there;
use OIDC for the operator routes a distribution adds. Do not advertise "OIDC-guarded"
for the data plane.

## CVE response

Base image is digest-pinned (`docker/Dockerfile`). On a base CVE: bump the digest
(`docker pull python:3.12-slim && docker inspect --format='{{index .RepoDigests 0}}'`),
rebuild, redeploy. The registry is stateless (rebuilds from heartbeats in ~5s), so
a rolling replace is safe.
