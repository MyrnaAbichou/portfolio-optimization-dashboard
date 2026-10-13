"""Step 4 - AI analyst: turns the numbers into a plain-language risk review.

The optimiser gives weights; it does not say *why* they look the way they do or
what could go wrong. Here a large language model (Claude, via the Anthropic API)
reads the results and writes a short critical commentary.

If no API key is set, a simple rule-based summary is returned instead, so the
rest of the project still works.
"""

import os

DEFAULT_MODEL = "claude-sonnet-5-5"


def build_prompt(asset_stats, max_sharpe, min_vol, perf_sharpe, perf_minvol,
                 correlations, risk_free_rate):
    """Package every number the model needs into one clear prompt."""
    def weights_text(w):
        return "\n".join(f"  - {t}: {v:.1%}" for t, v in w.sort_values(ascending=False).items()
                         if v > 0.005)

    return f"""You are a portfolio risk analyst reviewing a Modern Portfolio Theory study
built on historical daily data. Risk-free rate used: {risk_free_rate:.1%}.

INDIVIDUAL ASSETS (annualised):
{asset_stats.round(3).to_string()}

CORRELATION MATRIX:
{correlations.round(2).to_string()}

MAX-SHARPE PORTFOLIO - return {perf_sharpe[0]:.1%}, volatility {perf_sharpe[1]:.1%}, Sharpe {perf_sharpe[2]:.2f}
{weights_text(max_sharpe)}

MIN-VOLATILITY PORTFOLIO - return {perf_minvol[0]:.1%}, volatility {perf_minvol[1]:.1%}, Sharpe {perf_minvol[2]:.2f}
{weights_text(min_vol)}

Write a concise review (max ~250 words) with these sections:
1. What drives each portfolio's weights (refer to returns, volatility and correlations).
2. Main risks: concentration, assets with large drawdowns, dependence on the past period.
3. One concrete improvement to the methodology.
Be specific and critical. This is an educational exercise, not investment advice."""


def rule_based_summary(asset_stats, max_sharpe, perf_sharpe, perf_minvol):
    """Fallback analysis without AI: a few simple, transparent rules."""
    top = max_sharpe.idxmax()
    lines = [
        f"The max-Sharpe portfolio puts its largest weight in {top} "
        f"({max_sharpe[top]:.0%}).",
        f"It targets {perf_sharpe[0]:.1%} return for {perf_sharpe[1]:.1%} volatility "
        f"(Sharpe {perf_sharpe[2]:.2f}); the min-volatility portfolio cuts risk to "
        f"{perf_minvol[1]:.1%} for {perf_minvol[0]:.1%} return.",
    ]
    if max_sharpe.max() > 0.40:
        lines.append("Warning: more than 40% sits in one asset - consider a weight cap.")
    worst = asset_stats["Max drawdown"].idxmin()
    lines.append(f"Largest historical drawdown: {worst} "
                 f"({asset_stats.loc[worst, 'Max drawdown']:.0%}).")
    lines.append("Results rely on past returns, which are a noisy guide to the future.")
    return "\n\n".join(lines)


def ai_commentary(prompt, api_key=None, model=DEFAULT_MODEL):
    """Send the prompt to Claude and return its answer as text."""
    import anthropic  # imported here so the project runs without the package

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("No ANTHROPIC_API_KEY found.")
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
