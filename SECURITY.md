# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Report privately through GitHub: on the repository's **Security** tab choose
**Report a vulnerability**. Include what you found, how to reproduce it, and the impact you
expect. You should receive an acknowledgment within a week.

## Supported versions

Only the latest release on `main` receives security fixes.

## Scope

Relevant reports include:

- Code paths that could leak credentials (Alpaca keys, SEC contact details) into logs,
  result files, or commits
- Order-handling logic in `erised.execution` that could submit unintended trades
- Unsafe handling of downloaded data (for example, deserialization of untrusted input)
- Dependency vulnerabilities that are reachable through Erised

Trading losses, strategy performance, and data-vendor accuracy are out of scope.

## Handling credentials

Erised never needs credentials checked into the repository.

- Alpaca keys are read from `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. Use paper-trading keys
  unless you deliberately intend to trade real money.
- The SEC contact string is read from `SEC_EDGAR_USER_AGENT`.
- Put local values in `.env` (git-ignored; start from `.env.example`) or your shell profile.

If you accidentally commit a secret, revoke or rotate it with the provider immediately.
Rewriting git history afterwards does not make the leaked value safe again.
