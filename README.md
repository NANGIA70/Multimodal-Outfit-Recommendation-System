# Multimodal Outfit Recommendation System
An AI-powered personal styling assistant that combines computer vision, NLP, and retrieval-augmented generation (RAG) to recommend outfits based on clothing detection and contextual insights.

## Features
Clothing Detection: Utilizes Faster R-CNN with ResNet50 for fine-grained attribute classification of clothing items.

Context Processing: Employs OpenAI Whisper for audio-to-text transcription and NLP to process user preferences.

Tailored Recommendations: Integrates a RAG pipeline using Weaviate and Mistral-7B LLM to generate personalized outfit suggestions.

## Dataset
Fashionpedia (Hugging Face):

46,781 images with 342,182 bounding boxes.

Rich annotations covering 46 clothing categories with segmentation masks and hierarchical metadata.

Ideal for fine-grained detection tasks, outperforming alternatives like Fashion-MNIST and DeepFashion.

## Methodology
Computer Vision: Detects clothing attributes using Faster R-CNN with ResNet50 backbone.

NLP Pipeline: Processes fashion context via OpenAI Whisper and stores insights in a Weaviate vector database.

RAG Pipeline: Combines user queries with database knowledge using Mistral-7B LLM for personalized recommendations.

## Installation
Prerequisites
Python 3.8+

Docker (for containerized deployment)

## Steps
Clone the repository:
git clone https://github.com/NANGIA70/Multimodal-Outfit-Recommendation-System.git  
cd Multimodal-Outfit-Recommendation-System  

Run the application:
docker-compose up --build  

## Usage
Upload an image of your outfit or provide audio input describing your fashion preferences (e.g., occasion, aesthetic).

The system processes the input through the pipeline and returns tailored outfit recommendations based on detected attributes and contextual insights.

## Contributing
We welcome contributions! Please create a pull request or open an issue if you have suggestions or improvements.

