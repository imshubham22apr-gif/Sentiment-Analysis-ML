# pip install gradio transformers
import gradio as gr
from transformers import pipeline

# load the pre-trained sentiment analysis model for Hinglish
sentiment_analyzer = pipeline("sentiment-analysis", model="shae2977/xlm-roberta-hinglish-sentiment-analysis")

# define the function that will wrap our model
def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    # gradio works best with dictionary outputs
    return {result['label']: result['score']}

# create the Gradio Interface
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Type a sentence in Hinglish (e.g. yeh product bilkul bakwas hai)..."),
    outputs="label",
    title="Hinglish Sentiment Analysis Bot",
    description="Type in a Hinglish sentence to see if it's Positive, Negative or Neutral. Built with Gradio and Hugging Face Transformers."
)

if __name__ == "__main__":
    # launch the app!
    iface.launch(share=True)
