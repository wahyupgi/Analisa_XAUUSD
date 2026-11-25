# XAUUSD SMC ADAPTIVE AUTO TRADING SYSTEM — AGENT RULES

## 1. DEVELOPMENT RULES

- Prioritize accurate, minimal, modular implementation.
- Read only relevant files and direct dependencies.
- Use targeted code searches before opening files.
- Reuse findings already established in the current task.
- Fix verified root causes, not symptoms.
- Preserve SMC, SNR, MT5, news, journal, pantau, and all existing CLI features without regression.
- Reuse existing modules and avoid duplicated logic.
- Maintain a single source of truth for configuration, zone scoring, entry validation, and execution state.
- Never rewrite working features unnecessarily.
- Use Python scanner analysis as the sole execution authority.
- AI/LLM must never be an order decision maker.

## 2. CORE TRADING PRINCIPLES

- STRICT ON ZONE SELECTION.
- ADAPTIVE ON ENTRY EXECUTION.
- STRICT ON RISK MANAGEMENT.
- Prioritize fewer high-quality setups over excessive low-quality signals.
- Do not loosen zone quality merely to increase trading frequency.
- Do not introduce redundant confirmation filters that cause missed entries.
- Distinguish MARKET BIAS, TRADE PLAN, ENTRY STATUS, and EXECUTION STATUS.
- A valid trade plan is not automatically an executable entry.

## 3. ADAPTIVE ENTRY RULES

- Support EARLY_PENDING, ADAPTIVE_ENTRY, and CONFIRMED_ENTRY.
- ADAPTIVE_ENTRY is the primary execution mode.
- Use OR logic between independently validated entry triggers.
- One qualified trigger is sufficient when all mandatory validations pass.
- Supported triggers: Sweep + Reclaim, CHoCH, CISD, Rejection, Displacement, and Micro Structure Shift.
- Never require all entry triggers simultaneously.
- Do not require a new FVG when the existing zone is already valid.
- Allow neutral H1 when H4 and M30 support the trade direction.
- Block significant unresolved HTF directional conflicts.
- Never use stale triggers or triggers belonging to unrelated zones.
- Execute immediately when ENTRY_READY is reached in authorized FULL_AUTO mode.
- Never return WAIT solely because optional confirmations are missing.
- Never chase price outside the permitted execution range.
- Every WAIT or BLOCKED decision must provide a specific reason code.

## 4. ZONE QUALITY RULES

- Use existing SMC/ICT and SNR zone detection engines.
- Validate zone freshness, mitigation, structural invalidation, and directional alignment.
- Never execute invalidated, expired, or structurally broken zones.
- H4/H1 zones are not automatically strong solely because of their timeframe.
- Strong zones require objective structural evidence and valid confluence.
- Never manipulate zone scores to force an entry or larger lot.
- Initial scoring thresholds must be configurable and validated through testing.

## 5. DYNAMIC LOT MANAGEMENT

- Support real IDC/cent accounts using actual MT5 account currency.
- Use dynamic lot sizing based on validated zone quality.

Default lot classification:
- Score below 70: BLOCK.
- Score 70–79: maximum 0.01 lot.
- Score 80–89: maximum 0.02 lot.
- Score 90–100: maximum 0.05 lot only when Strong Zone Validation passes.

Additional rules:
- Strong H4/H1 Supply or Demand: maximum 0.05 lot.
- H4 + H1 confluence receives priority.
- M30/M15/M5 zones: maximum 0.02 lot.
- EARLY_PENDING: maximum 0.01 lot.
- High score without Strong Zone Validation: maximum 0.02 lot.
- Never exceed 0.05 lot.
- Target lot is a maximum allowance, not a mandatory volume.
- Calculate actual risk using mt5.order_calc_profit().
- Never hardcode USD/IDC currency conversion.
- Reduce lot from 0.05 to 0.02 to 0.01 when risk exceeds budget.
- If 0.01 exceeds risk budget, BLOCK execution.
- Never tighten SL artificially to allow a larger lot.

## 6. RISK MANAGEMENT RULES

Default configurable limits:
- EARLY_PENDING risk: 0.25% equity.
- ADAPTIVE_ENTRY risk: 0.5% equity.
- CONFIRMED_ENTRY risk: 0.5% equity.
- Absolute maximum risk per trade: 1%.
- Maximum total exposure: 1%.
- Maximum daily loss: 2%.
- Maximum active scanner positions: 1.
- Maximum active scanner pending orders: 1.
- Maximum trades per day: 3.
- Minimum actual RR: 1:1.5.

Mandatory rules:
- Use the lowest applicable risk limit.
- Calculate risk from actual equity, entry, SL, and broker specifications.
- Include commission and slippage buffers.
- Account for existing positions and pending order exposure.
- Require valid SL and TP for every order.
- Place SL beyond structural invalidation.
- Never widen SL after execution.
- Never use martingale, grid trading, averaging loss, or revenge trading.
- Never increase lot to recover previous losses.
- Persist daily loss and trade counters across restarts.
- Never bypass risk limits to force an entry.

## 7. REAL ACCOUNT IDC SAFETY

- Support direct execution on the currently connected real IDC account.
- Never hardcode login, password, server, or symbol.
- Never switch accounts automatically.
- Real account FULL_AUTO activation always requires explicit CLI command `auto real` and explicit user confirmation.
- AUTO_TRADING must be disabled by default.
- Verify account identity, currency, trade mode, equity, and margin before activation.
- Stop new execution immediately if account identity changes.
- Never place, modify, or cancel LIVE/REAL MT5 orders during development or automated testing.
- Never automatically resume real trading after restart without explicit authorization.
- Never modify or cancel manual orders or positions belonging to the user.
- Enforce scanner magic number, persistent setup ID, and verified order ownership.
- Account for manual position risk without modifying manual positions.

## 8. MT5 EXECUTION RULES

- Support BUY MARKET, SELL MARKET, BUY LIMIT, and SELL LIMIT.
- Use actual broker symbol specifications.
- Validate tick size, digits, volume step, filling mode, stops level, freeze level, and margin.
- Use Ask for BUY MARKET and Bid for SELL MARKET.
- Revalidate price, spread, SL, TP, RR, and risk immediately before execution.
- Always run mt5.order_check() before mt5.order_send().
- Verify broker retcode and actual execution result.
- Never report execution success without broker confirmation.
- Handle partial fills, rejection, timeout, requote, and ambiguous execution results.
- Reconcile orders, positions, and history before retrying.
- Never perform blind retries.
- Prevent duplicate orders and duplicate setup execution.
- Use thread-safe execution locking.

## 9. REAL-TIME MONITORING RULES

- Execution must not depend on manually entering the scan command.
- Monitor live ticks using a configurable interval, default 1 second.
- Refresh market structure when new candles form.
- Use closed candles for candle-confirmed triggers.
- Use live ticks for zone touch and execution price.
- Avoid look-ahead bias and repainting.
- Keep execution independent from CLI rendering and AI analysis.
- Stop new execution when market data is stale or MT5 disconnects.
- Never execute when market is closed.
- Preserve pre-market analysis and candidate zones when market is closed.

## 10. ORDER & POSITION MANAGEMENT

- Automatically cancel scanner pending orders when zones become invalidated or expired.
- Revalidate pending orders when market structure or risk conditions change.
- Cancel scanner pending orders before configured high-impact news blackout windows.
- Never create duplicate exposure when switching from pending to market execution.
- Verify cancellation and reconcile possible fills before replacement.
- Verify actual SL and TP after every fill.
- Apply emergency protection if a scanner position has no valid SL.
- Break-even and trailing stop must be optional and disabled by default.
- Never close an active position solely because a new zone appears.
- Never reopen the same stopped-out setup without fresh validation.
- Never modify manual user orders or positions.
- Do not assume server-side pending orders disappear when Python disconnects.

## 11. MANDATORY EXECUTION BLOCKERS

Block execution when:
- Zone is invalidated, expired, or structurally broken.
- Significant unresolved HTF conflict exists.
- SL or TP is missing or invalid.
- Actual RR is below minimum.
- Spread exceeds configured limits.
- Market data is stale.
- MT5 is disconnected.
- Market is closed.
- Algorithmic trading is disabled.
- High-impact news blackout is active.
- Required news data is unavailable or stale.
- Daily loss limit is reached.
- Risk or exposure limit is exceeded.
- Minimum lot exceeds risk budget.
- Margin is insufficient.
- Duplicate setup or order exists.
- Price leaves the permitted execution range.
- Broker volume or execution requirements are invalid.
- Account identity changes.
- Real account execution is not explicitly authorized.

Optional indicators must not become automatic execution blockers.

## 12. STATE & RECOVERY RULES

Use consistent execution states:
- RADAR
- APPROACHING
- ZONE_TOUCHED
- WAIT_TRIGGER
- ENTRY_READY
- EXECUTING
- ORDER_PLACED
- POSITION_ACTIVE
- COMPLETED
- INVALIDATED
- EXPIRED
- BLOCKED

- Persist setup state and execution identifiers.
- Reconcile MT5 orders, positions, and history after restart.
- Restore daily risk and trade counters.
- Prevent duplicate execution after reconnect or restart.
- Never assume an ambiguous execution failed.
- Never leave a fully validated setup in WAIT.

## 13. CLI & PANTAU RULES

- Preserve all existing CLI commands.
- Support auto status, auto on, auto off, auto real, auto demo, auto dryrun, auto semi, auto positions, auto orders, auto journal, auto risk, and auto emergency.
- Display account mode, currency, connection status, and execution mode.
- Display zone quality, score, trigger, entry, SL, TP, and actual RR.
- Display target lot, final lot, and lot adjustment reason.
- Display specific WAIT/BLOCKED reason codes.
- Display actual broker execution results.
- Keep output concise and readable.
- Never display simulated values as actual execution results.
- Emergency commands must stop new execution and cancel scanner pending orders.
- Closing active positions requires an explicitly authorized policy or user confirmation.

## 14. JOURNAL & TESTING RULES

- Preserve existing journal functionality.
- Record setup ID, zone ID, entry mode, zone score, trigger, target lot, final lot, risk, actual entry, SL, TP, RR, broker retcode, tickets, and P/L.
- Record missed entries, blocked entries, and execution latency.
- Support mock-MT5 unit and integration tests.
- Support no-lookahead backtests.
- Support read-only live dry-run checks.
- Test adaptive entry against confirmed entry.
- Test dynamic lot downgrade and risk-limit blocking.
- Test duplicate protection, restart recovery, and ambiguous execution.
- Test real account authorization and manual order protection.
- Never send real orders during development or automated testing.
- Never claim profitability without sufficient testing evidence.

## 15. FINAL IMPLEMENTATION RULES

- Implement actual code changes, not just recommendations.
- Preserve existing working functionality.
- Avoid unnecessary modules, dependencies, and complexity.
- Keep all risk thresholds and scoring thresholds configurable.
- Report changed files, verified root causes, implementation details, test results, and unresolved issues.
- Do not silently change user-approved trading rules.
- Do not weaken mandatory safety rules to increase entry frequency.

FINAL PRINCIPLE:

SELECTIVE ZONES.
DECISIVE EXECUTION.
DYNAMIC LOT.
CONTROLLED RISK.
NO REDUNDANT WAIT.
NO FORCED ENTRY.
NO DUPLICATE EXECUTION.