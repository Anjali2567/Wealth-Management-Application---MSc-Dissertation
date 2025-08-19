from huggingface_hub import InferenceClient
from .config import HUGGINGFACE_API_KEY

# HuggingFaceService class provides methods to interact with Cohere models via Hugging Face
class HuggingFaceService:
    def __init__(self, model_name="CohereLabs/c4ai-command-r-plus"):
        """
        Initialize the HuggingFaceService class using Cohere via Hugging Face.
        :param model_name: The name of the Hugging Face model to use (default: "CohereLabs/c4ai-command-r-plus").
        """
        self.client = InferenceClient(
            provider="cohere",
            api_key=HUGGINGFACE_API_KEY  # API key for authentication
        )
        self.model_name = model_name

    def interpret_portfolio_results(self, weights, tickers, predicted_returns):
        """
        Use a Cohere model via Hugging Face to interpret portfolio results.
        :param weights: List of portfolio weights.
        :param tickers: List of asset tickers.
        :param predicted_returns: Predicted returns for the assets.
        :return: Explanation of portfolio results.
        """
        # Prepare a summary of portfolio optimization results
        portfolio_summary = "\n".join(
            [f"{ticker}: {weight:.2f}% (Predicted Return: {predicted_returns[ticker]:.2f})"
             for ticker, weight in zip(tickers, weights)]
        )
        
        # Create a prompt for the Cohere model
        prompt = f"""
        The following is a portfolio optimization result:
        {portfolio_summary}

        Please justify the above allocations and explain why certain assets might be overweighted or underweighted in the portfolio considering real-world factors.
        Provide a human-friendly summary of the optimization results.
        """

        # Generate explanation using the Cohere model
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Extract the explanation from the response
        explanation = response.choices[0].message["content"]
        return explanation
    
    def get_risk_profile(self, prompt):
        """
        Send a prompt to the Cohere model to determine the risk profile.
        :param prompt: User-provided prompt describing the questionnaire and answers.
        :return: Risk profile string.
        """
        # Generate risk profile using the Cohere model
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Extract the risk profile from the response
        risk_profile = response.choices[0].message["content"]
        return risk_profile