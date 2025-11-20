#!/usr/bin/env python3
import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Setup Test - VLM Pipeline\n",
                "\n",
                "This notebook tests the core components:\n",
                "- Data Loader (XML parsing)\n",
                "- VLM Providers (Local, OpenAI, Google, Together AI)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "import os\n",
                "sys.path.append('..')\n",
                "\n",
                "from src.data_loader import DataLoader\n",
                "from src.providers import get_provider\n",
                "from src.config import Config\n",
                "import pandas as pd\n",
                "from IPython.display import display, Markdown, Image\n",
                "import json"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Test Data Loader"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "loader = DataLoader()\n",
                "df = loader.load_annotations()\n",
                "\n",
                "display(Markdown(f'### ✅ Successfully loaded {len(df)} annotations'))\n",
                "\n",
                "display(Markdown('### Sample Data:'))\n",
                "display(df.head(10))\n",
                "\n",
                "if not df.empty:\n",
                "    sample_image_path = df.iloc[0]['image_path']\n",
                "    display(Markdown(f'### Sample Image Path: `{sample_image_path}`'))\n",
                "    \n",
                "    if os.path.exists(sample_image_path):\n",
                "        display(Markdown('### ✅ Image file exists'))\n",
                "        display(Image(sample_image_path, width=400))\n",
                "    else:\n",
                "        display(Markdown('### ⚠️ Image not found'))\n",
                "        alt_path = os.path.join('..', 'Data', 'extractedimages', df.iloc[0]['file_name'])\n",
                "        if os.path.exists(alt_path):\n",
                "            display(Markdown(f'Found at: `{alt_path}`'))\n",
                "            display(Image(alt_path, width=400))\n",
                "            sample_image_path = alt_path\n",
                "else:\n",
                "    sample_image_path = None"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Test Local Provider (Ollama)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if sample_image_path and os.path.exists(sample_image_path):\n",
                "    display(Markdown('### Testing Local Provider (Ollama)'))\n",
                "    \n",
                "    provider = get_provider('local')\n",
                "    prompt = '''Analyze this property image and return a JSON object with:\n",
                "    {\n",
                "        \"is_clear\": boolean,\n",
                "        \"house_visible\": boolean,\n",
                "        \"description\": string\n",
                "    }'''\n",
                "    \n",
                "    display(Markdown(f'**Model:** {provider.model_name}'))\n",
                "    display(Markdown(f'**API URL:** {provider.api_url}'))\n",
                "    \n",
                "    try:\n",
                "        response = provider.analyze(sample_image_path, prompt)\n",
                "        if response:\n",
                "            display(Markdown('### ✅ Response Received:'))\n",
                "            try:\n",
                "                clean_response = response.strip()\n",
                "                if '```json' in clean_response:\n",
                "                    clean_response = clean_response.split('```json')[1].split('```')[0].strip()\n",
                "                elif '```' in clean_response:\n",
                "                    clean_response = clean_response.split('```')[1].split('```')[0].strip()\n",
                "                parsed = json.loads(clean_response)\n",
                "                display(Markdown('**Formatted JSON:**'))\n",
                "                display(json.dumps(parsed, indent=2))\n",
                "            except:\n",
                "                display(Markdown('**Raw Response:**'))\n",
                "                display(Markdown(f'```\\n{response}\\n```'))\n",
                "        else:\n",
                "            display(Markdown('### ❌ Empty response - Check if Ollama is running'))\n",
                "    except Exception as e:\n",
                "        display(Markdown(f'### ❌ Error: {str(e)}'))\n",
                "else:\n",
                "    display(Markdown('### ⚠️ Skipping: No valid image found'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. Test OpenAI Provider"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if sample_image_path and os.path.exists(sample_image_path) and Config.OPENAI_API_KEY:\n",
                "    display(Markdown('### Testing OpenAI Provider'))\n",
                "    provider = get_provider('openai')\n",
                "    prompt = 'Describe this property image briefly.'\n",
                "    display(Markdown(f'**Model:** {provider.model_name}'))\n",
                "    try:\n",
                "        response = provider.analyze(sample_image_path, prompt)\n",
                "        if response:\n",
                "            display(Markdown('### ✅ Response:'))\n",
                "            display(Markdown(response))\n",
                "        else:\n",
                "            display(Markdown('### ❌ Empty response'))\n",
                "    except Exception as e:\n",
                "        display(Markdown(f'### ❌ Error: {str(e)}'))\n",
                "else:\n",
                "    if not Config.OPENAI_API_KEY:\n",
                "        display(Markdown('### ⚠️ Skipping: OPENAI_API_KEY not set'))\n",
                "    else:\n",
                "        display(Markdown('### ⚠️ Skipping: No valid image found'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Test Google Provider"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if sample_image_path and os.path.exists(sample_image_path) and Config.GOOGLE_API_KEY:\n",
                "    display(Markdown('### Testing Google Provider'))\n",
                "    provider = get_provider('google')\n",
                "    prompt = 'Describe this property image briefly.'\n",
                "    display(Markdown(f'**Model:** {provider.model_name}'))\n",
                "    try:\n",
                "        response = provider.analyze(sample_image_path, prompt)\n",
                "        if response:\n",
                "            display(Markdown('### ✅ Response:'))\n",
                "            display(Markdown(response))\n",
                "        else:\n",
                "            display(Markdown('### ❌ Empty response'))\n",
                "    except Exception as e:\n",
                "        display(Markdown(f'### ❌ Error: {str(e)}'))\n",
                "else:\n",
                "    if not Config.GOOGLE_API_KEY:\n",
                "        display(Markdown('### ⚠️ Skipping: GOOGLE_API_KEY not set'))\n",
                "    else:\n",
                "        display(Markdown('### ⚠️ Skipping: No valid image found'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5. Test Together AI Provider"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if sample_image_path and os.path.exists(sample_image_path) and Config.TOGETHER_API_KEY:\n",
                "    display(Markdown('### Testing Together AI Provider'))\n",
                "    provider = get_provider('together')\n",
                "    prompt = 'Describe this property image briefly.'\n",
                "    display(Markdown(f'**Model:** {provider.model_name}'))\n",
                "    try:\n",
                "        response = provider.analyze(sample_image_path, prompt)\n",
                "        if response:\n",
                "            display(Markdown('### ✅ Response:'))\n",
                "            display(Markdown(response))\n",
                "        else:\n",
                "            display(Markdown('### ❌ Empty response'))\n",
                "    except Exception as e:\n",
                "        display(Markdown(f'### ❌ Error: {str(e)}'))\n",
                "else:\n",
                "    if not Config.TOGETHER_API_KEY:\n",
                "        display(Markdown('### ⚠️ Skipping: TOGETHER_API_KEY not set'))\n",
                "    else:\n",
                "        display(Markdown('### ⚠️ Skipping: No valid image found'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Summary\n",
                "\n",
                "All core components have been tested. Check the results above."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open('notebooks/00_setup_test.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
print('✅ Notebook created successfully')

