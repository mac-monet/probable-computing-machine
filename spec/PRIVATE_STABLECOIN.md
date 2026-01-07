# Private Stablecoin System

A privacy layer for stablecoins using client-side ZK proving with optimistic offline transfers and voluntary settlement.

## Goals

- Private transfers of any supported stablecoin (USDC, USDT, DAI, etc.)
- Client-side proving (no trusted prover)
- True offline transfers (no coordination required)
- Voluntary settlement (recipient chooses when to verify)
- No trusted mint (Chaumian ecash without the mint)

## Core Insight

ZK can prove computation correctness but cannot prove global state (e.g., "this nullifier was never used"). Rather than forcing synchronous settlement, we separate transfer from settlement:

- **Transfer**: Optimistic, instant, offline, P2P
- **Settlement**: Pessimistic, voluntary, online, when recipient chooses

The recipient bears double-spend risk until they settle. This is how physical cash works - you accept a bill optimistically, verify if you want certainty.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                       L1 (Ethereum)                          │
│                                                              │
│   Vault: deposit stablecoin → get private token              │
│          withdraw with proof → get stablecoin back           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Offline Transfer Layer                      │
│                                                              │
│   Tokens carry PCD proofs of valid history                   │
│   Transfers are instant, P2P, no coordination                │
│   Recipient bears risk until settlement                      │
└──────────────────────────┬──────────────────────────────────┘
                           │ voluntary
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Settlement Layer                          │
│                                                              │
│   Submit nullifier → get certainty                           │
│   Multiple options: rollup, validators, direct L1            │
│   Pay for finality when you want it                          │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Vault Contract (L1)

Holds stablecoin deposits. Accepts:

- Deposits: user sends stablecoin, gets commitment published
- Withdrawals: user proves ownership of unspent token, receives stablecoin

### Private Token

```
Token = {
    asset: which stablecoin
    commitment: hidden value and ownership
    accumulator: proof that all history is valid
    nullifier: unique identifier for settlement
}
```

The accumulator is a PCD proof - it grows by O(1) per transfer regardless of history length.

### P2P Transfers (Offline)

Sender proves:

1. They own the token (know the commitment opening)
2. The transfer is valid (amounts balance, rules satisfied)
3. History is valid (accumulator carries forward)

Receiver gets a new token with updated accumulator. No server involved. Transfer is instant but optimistic - receiver bears double-spend risk until settlement.

### Settlement Layer

Provides certainty that a token hasn't been double-spent. Multiple implementations possible:

**Option 1: Nullifier Rollup**

- Append-only nullifier set
- Periodic ZK proof of batch validity
- Anchors to L1
- Cheap, batched, some latency

**Option 2: Validator Certificates (Pod-style)**

- Threshold of validators (2/3 signatures)
- Each validator maintains local nullifier set
- Signs only first spend they see
- Fast (single round trip), no consensus needed

**Option 3: Direct L1**

- Submit nullifier directly to vault contract
- Maximum security, highest cost
- For high-value settlements

Settlement is a service, not a constraint. Users choose cost/speed/trust tradeoff.

## Flow

**Deposit**: Send stablecoin to vault → receive initial private token (settled by default)

**Transfer**: Prove ownership + valid transfer → recipient gets new token (instant, offline, optimistic)

**Settle** (voluntary): Submit nullifier to settlement layer → get certainty token isn't double-spent

**Withdraw**: Prove ownership of settled token to vault → receive stablecoin

## Risk Model

Recipient bears double-spend risk until settlement. Natural risk stratification emerges:

| Scenario | Behavior |
|----------|----------|
| Small payment ($5 coffee) | Maybe never settle. Risk is trivial. |
| Large payment ($2000 rent) | Settle within hours. Can't afford loss. |
| Merchant (many small payments) | Batch settle daily. Aggregate risk management. |
| High-trust parties (friends) | Settle lazily. Social trust substitutes. |

This mirrors physical cash: you accept bills optimistically, verify if the amount warrants it.

## Key Properties

**Money velocity exceeds settlement throughput**: The system can process far more transfers than the settlement layer handles, because most don't need immediate settlement.

**Graceful degradation**: If settlement layer is slow or down, money still moves. Just with more risk.

**Privacy gradient**: Offline transfers are maximally private. Settlement reveals nullifiers. Users choose their privacy/certainty tradeoff.

**No artificial deadlines**: Unlike time-locked schemes, tokens don't expire. Your risk accumulates, but the token stays valid.

## Privacy Properties

- **Sender privacy**: Nullifier reveals nothing about sender
- **Receiver privacy**: New commitment reveals nothing about receiver
- **Amount privacy**: Values hidden in commitments
- **Transaction graph**: Transfers are unlinkable (each token has fresh commitment)
- **Settlement timing**: User-controlled privacy/certainty tradeoff

## Trust Assumptions

- L1 security for vault contract
- Settlement layer for liveness (not safety - can't steal funds, only delay certainty)
- Client-side proving is correct (open source, verifiable)

No trusted mint. No party can inflate supply or link transactions. No hard coordination requirements.

## Why PCD

Traditional ZK (single SNARK) requires proving entire history in one shot. PCD allows:

- Incremental proving (each transfer adds O(1) work)
- Client-side proving without massive computation
- Flexible (can add features without new trusted setup)

## Theoretical Foundation

Cryptography can prove computation but not global state. Double-spend prevention requires knowing "this nullifier was never used anywhere" - a universal negative that requires global knowledge.

Rather than fight this limit, we embrace it:

- Proofs handle validity (what cryptography can do)
- Settlement handles uniqueness (what requires coordination)
- Users choose when to coordinate based on their risk tolerance

This is the most "offline" honest design possible. True offline finality without trust assumptions is information-theoretically impossible.

## Hardware Attestation (Optional Trust Layer)

Hardware secure enclaves (Apple Secure Enclave, Android StrongBox/TEE) can provide additional double-spend resistance without requiring settlement.

### How It Works

The device's secure enclave:

1. Maintains a local nullifier set (survives app reinstalls)
2. Refuses to sign a transfer if nullifier was already used on this device
3. Provides remote attestation that the signing code is legitimate

```
Token = {
    commitment,
    accumulator,           // PCD proof: valid history
    device_attestation,    // Hardware: "this device hasn't spent this"
}
```

### Trust Model

| Attack Vector | Without Attestation | With Attestation |
|---------------|---------------------|------------------|
| Same device double-spend | Possible | Blocked by enclave |
| Multiple devices | Possible | Still possible |
| Rooted/jailbroken | Possible | Detectable (attestation fails) |
| Compromised enclave | N/A | Possible but expensive |

Hardware attestation doesn't eliminate double-spend risk, but adds significant friction. Most casual attackers don't have multiple clean devices ready.

### Layered Certainty

Recipients can adjust acceptance thresholds based on attestation:

```
1. Raw proof only           → Low trust, small amounts
2. + Device attestation     → Medium trust, moderate amounts
3. + Multiple attestations  → Higher trust, larger amounts
4. + Settlement             → Full certainty, any amount
```

### Tradeoffs

**Benefits:**

- Instant additional trust without network round-trip
- Raises attack cost (need multiple devices)
- Graceful: works offline, enhances rather than replaces

**Costs:**

- Trust in hardware vendor (Apple/Google)
- Platform-specific implementation
- Privacy concern: attestation may leak device identity
- Doesn't help against determined attacker with multiple devices

### Privacy Considerations

Device attestations should be designed to avoid linking transactions:

- Use blind attestation schemes where possible
- Rotate attestation keys
- Attest to "not double-spent" without revealing which device

## Viewing Keys (Selective Disclosure)

Viewing keys enable compliance and auditability without sacrificing privacy by default. Users can selectively reveal transaction history to authorized parties while retaining exclusive spending control.

### Key Derivation

```
master_seed
    │
    ├── spending_key     Can transfer tokens (KEEP SECRET)
    │
    └── viewing_key      Can see transaction history (SHAREABLE)
          │
          ├── incoming_viewing_key    See received transactions
          └── full_viewing_key        See all transactions + balances
```

The viewing key is cryptographically derived from the spending key, but the reverse is computationally infeasible. Sharing your viewing key cannot compromise your funds.

### What Viewing Keys Reveal

| Key Type | Reveals | Use Case |
|----------|---------|----------|
| Incoming viewing key | Deposits received, amounts, timestamps | Payment verification |
| Full viewing key | All transactions, current balance, counterparties | Full audit |
| Spending key | Everything + ability to spend | Never share |

### Compliance Use Cases

**Tax Reporting:**

```
User → Tax Authority: full_viewing_key for relevant period
Authority: Can verify income, capital gains, holdings
Authority: Cannot spend funds or see other accounts
```

**Business Audit:**

```
Company → Auditor: full_viewing_key for company wallet
Auditor: Verify all transactions, reconcile books
Auditor: Cannot move funds
```

**Payment Verification:**

```
Sender → Recipient: proof of payment (single transaction disclosure)
Recipient: Verify payment was made
Recipient: Cannot see sender's other transactions
```

**Legal Discovery:**

```
Court order → User must provide viewing_key
Scoped: Only for relevant addresses/time periods
User: Retains spending control
```

### Implementation with PCD

The viewing key is embedded in the token structure:

```
Token = {
    commitment: hide(value, owner_pubkey, blinding),
    accumulator: PCD proof of valid history,
    viewing_tag: encrypt(viewing_key, transaction_metadata),
}

Metadata encrypted under viewing_key:
    - amount
    - timestamp
    - counterparty (if known)
    - memo (optional)
```

Anyone with the viewing key can decrypt the metadata. Without it, the transaction is opaque.

### Selective Disclosure Proofs

For finer-grained disclosure, generate ZK proofs about transactions without revealing full history:

```
"I received at least $10,000 in Q4 2025"
  → Proves aggregate without itemizing

"This payment of $500 was made to address X"
  → Proves single transaction

"My balance is above $1,000"
  → Proves solvency without exact amount

"All my deposits came from compliant sources"
  → Proves clean history without revealing sources
```

These use the viewing key to access data, then generate a ZK proof over that data.

### Key Rotation

Viewing keys can be rotated without moving funds:

```
1. Generate new viewing_key' from spending_key
2. New transactions encrypted to viewing_key'
3. Old viewing_key still decrypts historical transactions
4. Compartmentalize: different auditors see different periods
```

### Trust Model

| What | Who Knows |
|------|-----------|
| Transaction occurred | Only parties + anyone with viewing key |
| Transaction details | Only with viewing key |
| Spending ability | Only spending key holder |
| Full history | Only full viewing key holder |

**Privacy by default.** Disclosure is opt-in, scoped, and doesn't compromise spending authority.

### Comparison to Other Systems

| System | Disclosure Mechanism |
|--------|---------------------|
| Bitcoin | Fully transparent (no privacy) |
| Monero | View keys (similar to this) |
| Zcash | [Viewing keys](https://electriccoin.co/blog/explaining-viewing-keys-2/) (this design inspired by Zcash) |
| Tornado Cash | No disclosure mechanism (contributed to OFAC sanctions) |
| This system | Viewing keys + selective ZK proofs |

## Open Design Questions

- Settlement layer decentralization (single operator vs federation vs permissionless)
- Withdrawal privacy (batching, timing attacks)
- Fee mechanism (who pays for settlement, how)
- Multi-chain support (same tokens across L1s)
- Economic deterrents for double-spenders (bonds, reputation, traitor tracing)
- Hardware attestation privacy (blind attestations, key rotation)
- Alternative trust signals beyond hardware

---

## Application: Streaming Micropayments

The optimistic transfer model enables high-throughput micropayments where traditional systems fail. Example: paying for torrent data chunk-by-chunk.

### The Challenge

| Operation | Latency |
|-----------|---------|
| Sign a message | ~1ms |
| Generate PCD proof | ~1-10 seconds |

Generating a full PCD proof per chunk bottlenecks at <1 chunk/second. Torrents need thousands/second.

### Solution: Separate Acknowledgment from Settlement

Stream fast signed receipts during the session. Batch into a single PCD proof at settlement.

```
┌─────────────────────────────────────────────────────────────┐
│  Session Setup                                              │
├─────────────────────────────────────────────────────────────┤
│  Leecher: commit to max_payment M                           │
│  Seeder: agree to serve up to N chunks                      │
│  Exchange: session keys for fast signing                    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Streaming (per chunk, ~1ms each)                           │
├─────────────────────────────────────────────────────────────┤
│  Seeder → Leecher: chunk[i]                                 │
│  Leecher → Seeder: sig(session_id, counter=i, chunk_hash)   │
│                                                             │
│  No ZK proofs. Just signatures. Seeder accumulates IOUs.    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Settlement (once, at session end)                          │
├─────────────────────────────────────────────────────────────┤
│  Leecher: generate PCD proof for final amount               │
│  Seeder: receives private token                             │
│  One proof covers entire session (thousands of chunks)      │
└─────────────────────────────────────────────────────────────┘
```

Throughput limited only by signing speed (~1000s/sec), not proof generation.

### Handling Non-Settlement

If the leecher disappears without settling:

**Upfront bond**: Leecher locks collateral at session start. No settlement → seeder claims bond with signed receipts as evidence.

**Incremental settlement**: Every N chunks, leecher provides intermediate proof. Seeder stops serving if proof is late. Risk bounded to N chunks.

**Reputation**: Leechers build reputation across sessions. New leechers get small credit limits. Known leechers get larger limits.

**Probabilistic payments**: Only some chunks require real proofs (see below).

### Payment Channels (Replace-by-Incentive)

For streaming payments, unidirectional payment channels offer exact payments with minimal overhead. This uses the "replace-by-incentive" pattern: since the receiver always prefers higher balances, they naturally keep only the latest (highest) signed state. No nonces, no state machine, no disputes.

#### Security Model

A naive payment channel (just exchanging signatures referencing a token) is vulnerable:

```
Attack: Double-spend the underlying token
1. Alice opens channel with Bob using token T
2. Alice streams payments to Bob ($50 accumulated)
3. Alice transfers T to Carol (separate transaction)
4. Carol settles, T's nullifier is spent
5. Bob tries to close channel → fails (T already spent)
6. Bob loses $50
```

**Solution: Escrow on the settlement layer.** When opening a channel, the token is locked in escrow. It cannot be transferred or used in another channel until the channel closes.

#### Protocol with Escrow

```
┌─────────────────────────────────────────────────────────────┐
│  Open Channel (requires settlement layer)                   │
├─────────────────────────────────────────────────────────────┤
│  Leecher → Rollup: lock(token_T, channel_id, seeder, 24h)   │
│  Rollup: marks T as locked, records channel parameters      │
│  T cannot be transferred or used elsewhere until close      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stream Payments (fully offline, ~1ms each)                 │
├─────────────────────────────────────────────────────────────┤
│  Chunk 1: leecher signs "seeder can claim ≤ $0.001 from T"  │
│  Chunk 2: leecher signs "seeder can claim ≤ $0.002 from T"  │
│  Chunk 3: leecher signs "seeder can claim ≤ $0.003 from T"  │
│  ...                                                        │
│  Chunk N: leecher signs "seeder can claim ≤ $4.50 from T"   │
│                                                             │
│  Seeder just keeps highest. No rollup interaction.          │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Close Channel (two options)                                │
├─────────────────────────────────────────────────────────────┤
│  Option A - Seeder closes:                                  │
│    Seeder → Rollup: close(channel_id, $4.50, signature)     │
│    Rollup verifies signature, releases:                     │
│      - $4.50 → seeder (new token)                           │
│      - $5.50 → leecher (remainder token)                    │
│                                                             │
│  Option B - Timeout refund:                                 │
│    After 24h, if seeder hasn't closed:                      │
│    Leecher → Rollup: refund(channel_id)                     │
│    Rollup returns full T to leecher                         │
└─────────────────────────────────────────────────────────────┘
```

#### Escrow API

The settlement layer adds three functions:

```
lock(token_proof, channel_id, counterparty, timeout)
  → Verifies token ownership
  → Marks token as locked to this channel
  → Records: channel_id, owner, counterparty, timeout

close(channel_id, amount, signature)
  → Verifies signature from owner authorizing amount
  → Splits locked token: amount → counterparty, remainder → owner
  → Removes channel

refund(channel_id)
  → Only callable after timeout
  → Returns full token to owner
  → Removes channel
```

#### Security Properties

| Attack | Prevented by |
|--------|--------------|
| Transfer T while channel open | Rollup rejects (T is locked) |
| Open multiple channels with T | Rollup rejects (T already locked) |
| Seeder claims more than signed | Can't forge owner's signature |
| Seeder disappears | Owner reclaims after timeout |
| Owner posts old state | Can't - only counterparty closes with signatures |

#### Why Replace-by-Incentive Still Works

The seeder always keeps the highest signed amount. No nonces needed:

| Party | Can they cheat? | Why not? |
|-------|-----------------|----------|
| Leecher | No | Token is locked; can't double-spend |
| Seeder | No | Can only claim what leecher signed |

The seeder is always incentivized to use the highest signed amount. Incentives replace enforcement.

#### Comparison

| | Lightning | This Design |
|-|-----------|-------------|
| Open channel | On-chain tx | Rollup escrow |
| State updates | Both sign, complex | Only payer signs, simple |
| Disputes | Yes (penalty) | No (unidirectional) |
| Close | On-chain tx | Rollup release |
| Streaming | Offline | Offline |

#### Connection to Overall Design

The system offers two models for different use cases:

```
Regular transfers:  Optimistic, fully offline
                    Recipient bears risk until they settle
                    Best for: one-off payments, small amounts

Payment channels:   Escrow-secured, offline streaming
                    No risk after channel opens
                    Best for: ongoing relationships, high volume
```

Both rely on rational self-interest. Regular transfers trust recipients to settle when they need certainty. Channels trust that receivers want the highest balance.

**Historical note:** The replace-by-incentive pattern originates from Spillman channels (2013), the earliest Bitcoin payment channel design. This design adapts it with rollup escrow for security while preserving the simplicity of unidirectional state updates.

### Probabilistic Micropayments

Instead of proving every micropayment, use lottery-based payments:

```
Setup:
  price_per_chunk = $0.0001
  payment_amount = $0.10
  probability = 0.1%

Per chunk:
  ticket = hash(chunk_content || chunk_id || session_entropy)
  if ticket < threshold:
    → "Winning" chunk: generate real PCD payment proof
    → Seeder receives $0.10
  else:
    → Just signed receipt, no proof needed

Expected value: 0.1% × $0.10 = $0.0001 per chunk ✓
```

Benefits:

- 99.9% of chunks: fast signatures only
- 0.1% of chunks: full PCD proof
- Correct expected payment
- Variance negligible over thousands of chunks
- Massive throughput improvement

#### Verifiable Fairness

A naive lottery is gameable - if either party can predict winners, they cheat:

| If seeder knows winners | If leecher knows winners |
|-------------------------|--------------------------|
| Drop winning chunks | Only request losing chunks |
| Only sends losers | Pays nothing |
| Gets paid nothing | |

**Solution: Neither party can bias the lottery.**

For torrents, chunk content is predetermined (part of the file). Use it as entropy:

```
┌─────────────────────────────────────────────────────────────┐
│  Session Setup                                              │
├─────────────────────────────────────────────────────────────┤
│  Seeder: S_commit = hash(seeder_secret)                     │
│  Leecher: L_commit = hash(leecher_secret)                   │
│  Exchange commitments, then both reveal                     │
│  session_entropy = hash(seeder_secret || leecher_secret)    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Per Chunk                                                  │
├─────────────────────────────────────────────────────────────┤
│  Seeder → Leecher: chunk_content                            │
│  ticket = hash(chunk_content || chunk_index || session_entropy)  │
│                                                             │
│  if ticket < threshold:                                     │
│    Winner! Leecher generates PCD proof for $0.10            │
│  else:                                                      │
│    Leecher sends signed receipt (no proof)                  │
└─────────────────────────────────────────────────────────────┘
```

**Why neither party can cheat:**

| Party | Attack | Why it fails |
|-------|--------|--------------|
| Seeder | Drop winning chunks | Can't predict - doesn't know leecher_secret until committed |
| Seeder | Change chunk content | Content is fixed by the file/torrent |
| Leecher | Request only losers | Can't predict - doesn't know chunk content until received |
| Leecher | Lie about lottery result | PCD proof must include chunk_hash, verifiable |

#### PCD Integration

The proof for a winning chunk proves the lottery was fair:

```
WinningChunkProof = PCD.prove(
    public: {
        chunk_hash,
        chunk_index,
        session_entropy_commitment,
        payment_amount,
    },
    private: {
        chunk_content,
        session_entropy,
    },
    constraints: {
        hash(chunk_content) == chunk_hash,
        hash(session_entropy) == session_entropy_commitment,
        hash(chunk_content || chunk_index || session_entropy) < threshold,
        // ... accumulator update, payment validity
    }
)
```

The verifier confirms:

1. Chunk content matches claimed hash
2. Session entropy matches committed value
3. Lottery result is correctly computed
4. Payment proof is valid

#### For Non-Predetermined Content

If chunk content isn't fixed (arbitrary data streams, not torrents), use chained randomness:

```
R_0 = session_entropy
R_i = hash(R_{i-1} || chunk_hash_{i-1})

Lottery for chunk i uses R_i
```

Each chunk's randomness depends on the previous chunk's content. Neither party can look ahead - you must receive chunk i-1 to know the lottery for chunk i.

Alternative: Mini commit-reveal per chunk (higher overhead):

```
1. Leecher: sends R_commit = hash(R)
2. Seeder: sends chunk
3. Leecher: reveals R
4. Winner = hash(chunk_content || R) < threshold

Leecher committed before seeing chunk (can't grind)
Seeder sent chunk before seeing R (can't selectively drop)
```

### Throughput Summary

| Layer | Mechanism | Speed |
|-------|-----------|-------|
| Per-chunk | Signed receipts | ~1ms |
| Winning chunks | Probabilistic PCD proofs | ~1-10s (rare) |
| End of session | Final settlement proof | ~1-10s (once) |
| Settlement layer | Seeder settles when ready | Async |

### Academic References

Probabilistic micropayments have been studied extensively:

- Rivest, R. L. (1997). "Electronic Lottery Tickets as Micropayments." *Financial Cryptography (FC '97)*. The original lottery-based micropayment proposal.

- Wheeler, D. (1996). "Transactions Using Bets." *Security Protocols Workshop*. Early theoretical foundation.

- Micali, S., & Rivest, R. L. (2002). "Micropayments Revisited." *CT-RSA 2002*. Formalized probabilistic payments with efficient coin-flipping.

- Pass, R., & shelat, a. (2015). "Micropayments for Decentralized Currencies." *ACM CCS 2015*. Adapted probabilistic payments for cryptocurrency settings.

- Chiesa, A., Green, M., Liu, J., Miao, P., Miers, I., & Mishra, P. (2017). "Decentralized Anonymous Micropayments." *EUROCRYPT 2017*. Combines probabilistic payments with zero-knowledge proofs.

The model presented here extends these ideas with PCD accumulators for private, offline-capable streaming payments.
