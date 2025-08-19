import streamlit as st
from main import predict_returns, optimize_portfolio
from services.execute_trades import execute_trades, check_rebalance_needed, get_last_working_day, get_asset_names_from_alpaca
from services.huggingface_service import HuggingFaceService
# import os


st.set_page_config(page_title="Wealth Management Portfolio Optimizer", layout="centered")


st.markdown("<h1 style='font-size:30px;'>Welcome to Wealth Management Application</h1>", unsafe_allow_html=True)
st.markdown("Press predict to predict the future prices of assets.")
if st.button("Predict"):
    # Sector to ticker mapping
    st.session_state.sector_tickers = {
        "Technology": ["AAPL", "MSFT", "GOOGL"],
        "Healthcare": ["JNJ", "PFE", "MRK"],
        "Finance": ["JPM", "BAC", "WFC"],
        "Energy": ["XOM", "CVX", "COP"],
        "Consumer Goods": ["PG", "KO", "PEP"],
        "Utilities": ["NEE", "DUK", "SO"],
        "Industrials": ["HON", "UNP", "UPS"]
    }
    st.session_state.always_include = {"BND": "Bonds", "VNQ": "Real Estate", "GLD": "Gold"}

    st.session_state.all_sectors = list(st.session_state.sector_tickers.keys())
    all_tickers = list(st.session_state.always_include.keys())

    for sector in st.session_state.all_sectors:
        all_tickers.extend(st.session_state.sector_tickers[sector])
    st.session_state.all_tickers = all_tickers

    start_date_str = "2010-01-01"
    end_date_str = get_last_working_day()
    with st.spinner("Predicting future prices..."):
        predicted_returns, returns = predict_returns(
            st.session_state.all_tickers, start_date_str, end_date_str
        )
    st.session_state.predicted_returns = predicted_returns
    st.session_state.returns = returns


# Display predicted returns in a presentable way with asset names
if "predicted_returns" in st.session_state and st.session_state.predicted_returns is not None:
    asset_names = get_asset_names_from_alpaca(list(st.session_state.predicted_returns.keys()))
    st.markdown("### Predicted Biweekly Returns")
    st.table([
        {"Asset": asset_names.get(ticker, ticker), "Predicted Return (%)": round(return_value * 100, 2)}
        for ticker, return_value in st.session_state.predicted_returns.items()
    ])

# Add disclaimer
st.markdown(
    """
    <span style='color:red; font-weight:bold;'>
    Disclaimer: Investing in financial markets involves risk, including the loss of principal. Past performance is not indicative of future results.
    </span>
    """,
    unsafe_allow_html=True
)

st.markdown("""
Answer the following questions to determine your investment risk profile.
""")

# Define questions, choices, and scores
questions = [
    {
        "question": "How long do you plan to invest this money before needing it?",
        "choices": {
            "Less than 3 years": 1,
            "3-5 years": 2,
            "5-10 years": 3,
            "10-20 years": 4,
            "More than 20 years": 5
        }
    },
    {
        "question": "Which of the following best describes your investment objective?",
        "choices": {
            "Capital preservation with minimum potential value fluctuations and capital losses": 1,
            "Low but stable returns with low potential value fluctuations and capital losses": 2,
            "Moderate returns with medium potential value fluctuations and capital losses": 3,
            "High returns with high potential value fluctuations and capital losses": 4,
            "Maximum returns with very high potential value fluctuations and capital losses": 5
        }
    },
    {
        "question": "How would you react if your investment assets were to suffer an unrealised loss of 20%?",
        "choices": {
            "Sell all investments as you do not want to take the risk of further losses": 1,
            "Sell the majority of your investments to minimise risks of further losses": 2,
            "Sell some but not the majority of your investments to reduce risk, while waiting for the remaining to recover in value": 3,
            "Hold on to the investments, with expectations for performance to improve": 4,
            "Buy additional investments as part of your portfolio to take advantage of lower prices": 5
        }
    },
    {
        "question": "How much loss would you be willing to tolerate before selling your investments?",
        "choices": {
            "Loss of up to 5%": 1,
            "Loss of up to 10%": 2,
            "Loss of up to 20%": 3,
            "Loss of up to 30%": 4,
            "Loss of more than 30%": 5
        }
    },
    {
        "question": "When faced with large immediate cash needs, to what extent would you need to rely on assets in this account?",
        "choices": {
            "You need to withdraw all assets": 1,
            "You need to make a large withdrawal": 2,
            "You need to make a small to medium withdrawal": 3,
            "You are unlikely to make withdrawals": 4,
            "You do not need to make withdrawals": 5
        }
    }
]

# Instantiate the HuggingFaceService
hf_service = HuggingFaceService()

if "responses" not in st.session_state:
    st.session_state.responses = [None] * len(questions)

for i, q in enumerate(questions):
    st.session_state.responses[i] = st.radio(q["question"], list(q["choices"].keys()), index=0, key=f"q{i}")

if "risk_pref" not in st.session_state:
    st.session_state.risk_pref = None
# Submit button for risk questionnaire
if st.button("Submit Risk Questionnaire"):
    # Prepare prompt for LLM
    answers = [f"Q{i+1}: {q['question']} A: {resp}" for i, (q, resp) in enumerate(zip(questions, st.session_state.responses))]
    prompt = (
        "Given the following answers to a risk assessment questionnaire, "
        "determine the user's investment risk profile as one of: "
        "'Very Conservative', 'Conservative', 'Moderate', 'Aggressive', or 'Very Aggressive'.\n\n"
        + "\n".join(answers) +
        "\n\n and give a 2-3 line explanation for why the specific profile." +
        "\n Risk Profile:"
    )

    # Call the LLM
    with st.spinner("Determining your risk profile using AI..."):
        risk_pref = hf_service.get_risk_profile(prompt).strip()
    st.session_state.risk_pref = risk_pref
    st.success(f"Risk profile determined")
st.subheader(f"{st.session_state.risk_pref}")


st.markdown("### Select your preferred sectors (check one or more):")
all_sectors = list(st.session_state.sector_tickers.keys())
no_pref = st.checkbox("No preference (selects all sectors)", key="no_pref_sector")
sector_choices = []

if no_pref:
    sector_choices = all_sectors
else:
    for sector in all_sectors:
        checked = st.checkbox(sector, key=f"sector_{sector}")
        if checked:
            sector_choices.append(sector)
st.session_state.sector_choices = sector_choices

if "show_portfolio" not in st.session_state:
    st.session_state.show_portfolio = False
if "explanations" not in st.session_state:
    st.session_state.explanations = ""

if st.button("Calculate Portfolio"):
    tickers = list(st.session_state.always_include.keys())
    for sector in st.session_state.sector_choices:
        tickers.extend(st.session_state.sector_tickers[sector])
    st.session_state.tickers = tickers

    with st.spinner("Optimizing portfolio..."):
        weights, explanations = optimize_portfolio(
            st.session_state.tickers, st.session_state.predicted_returns, st.session_state.returns, st.session_state.risk_pref
        )
    st.session_state.calculated_weights = weights
    st.session_state.explanations = explanations
    st.session_state.show_portfolio = True
    st.session_state.asset_names = get_asset_names_from_alpaca(list(st.session_state.calculated_weights.keys()))

if st.session_state.show_portfolio and st.session_state.calculated_weights is not None:
    st.table(
        [
            {
                "Asset": st.session_state.asset_names.get(asset, asset),
                "Allocation (%)": st.session_state.calculated_weights[asset]
            }
            for asset in st.session_state.calculated_weights.keys()
        ]
    )

    st.subheader(f"{st.session_state.explanations}")

# Ensure edit_mode persists across reruns
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False

if "edited_weights" not in st.session_state:
    st.session_state.edited_weights = {}

col1, col2, col3 = st.columns(3)
accept = col1.button("Accept Portfolio")
reject = col2.button("Reject Portfolio")
edit = col3.button("Edit Portfolio")

if edit:
    st.session_state.edit_mode = True

if reject:
    st.session_state.show_portfolio = False
    st.session_state.calculated_weights = None
    st.session_state.tickers = None
    st.session_state.edit_mode = False

if accept:
    st.info("Executing trades in Alpaca...")
    trade_results = execute_trades(st.session_state.calculated_weights)
    for msg in trade_results:
        st.write(msg)
    st.session_state.show_portfolio = False
    st.session_state.edit_mode = False

if st.session_state.edit_mode:
    st.markdown("#### You can manually adjust the allocations below (sum must be 100):")
    for asset, weight in st.session_state.calculated_weights.items():
        st.session_state.edited_weights[asset] = st.number_input(
            f"{st.session_state.asset_names.get(asset, asset)} Allocation (%)",
            min_value=0.0, max_value=100.0,
            value=float(st.session_state.edited_weights.get(asset, weight)),
            step=0.1,
            key=f"edit_{asset}"
        )

    total_edited = sum(st.session_state.edited_weights.values())
    st.write(f"**Total Allocation:** {total_edited:.2f}%")

    if st.button("Confirm Edited Portfolio"):
        if abs(total_edited - 100.0) > 0.01:
            st.error("The total allocation must sum to 100%. Please adjust the weights.")
        else:
            st.session_state.calculated_weights = {asset: round(w, 2) for asset, w in st.session_state.edited_weights.items()}
            st.success("Portfolio weights updated. You can now accept or reject this portfolio.")

            st.table(
                [
                    {
                        "Asset": st.session_state.asset_names.get(asset, asset),
                        "Allocation (%)": st.session_state.edited_weights[asset]
                    }
                    for asset in st.session_state.edited_weights.keys()
                ]
            )

            st.session_state.edit_mode = False
            st.session_state.edited_weights = {}

if st.button("Check Rebalance Need"):
    rebalance_needed, rebalance_assets = check_rebalance_needed(
        st.session_state.calculated_weights, threshold=5.0
    )

    asset_names = get_asset_names_from_alpaca([asset for asset in rebalance_assets.keys()])

    if rebalance_needed:
        st.warning("Some assets have drifted beyond the 5% threshold. Rebalancing is recommended.")
        st.table([
            {
                "Asset": asset_names.get(asset, asset),
                "Target Allocation (%)": vals["Target (%)"],
                "Current Allocation (%)": vals["Current (%)"],
                "Difference (%)": vals["Difference (%)"]
            }
            for asset, vals in rebalance_assets.items()
        ])
        if st.button("Accept Rebalance"):
            st.info("Executing rebalance trades in Alpaca...")
            trade_results = execute_trades(st.session_state.calculated_weights)
            for msg in trade_results:
                st.write(msg)
    else:
        st.success("All asset allocations are within the 5% threshold. No rebalancing needed.")