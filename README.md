# Experimental Results: Zero-Shot vs. SFT in Visual Object Counting

## 1. Introduction
This document compares Zero-Shot (without fine-tuning) vs. SFT (fine-tuned model) for the task of visual object counting. The goal is to assess how well the model performs in recognizing and counting objects in images before and after supervised fine-tuning (SFT). When viewing the solution, I advise you to open the version of notebook in Colab.

## 2. Methodology
### 2.1 Dataset
**Dataset**: [CLEVR-Cogen-A-Train](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct#how-to-get-started) (70.000 images)

**Data split**:
* Train – 3.500 images (5%)
* Validation – 350 images (0,5%)
* Test – 998 images (1,5%)

Splitting the data set (the size of the split was chosen experimentally so that Colab resources were sufficient and that training did not take too long). The dataset was shuffled before splitting to ensure randomness and avoid any bias.

### 2.2 Model and Setup
**Model**: [SmolVLM-256M-Instruct](https://huggingface.co/blog/smolvlm)
#### 2.2.1 Zero-Shot Setup:
The model receives an image and a prompt without any additional training.
Example prompt: 
```python
messages = [
{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": 'How many items are there in the image?'}
    ]
},
]
```

#### 2.2.2 Supervised Fine-Tuning (SFT) Setup:
In this notebook I will not do full fine-tuning, but used QLoRA. QLoRA is used instead of full fine-tuning to reduce memory usage, making it possible to fine-tune large models even on limited resources like Google Colab.

The model is fine-tuned on 3,500 images. Example prompt: 
```python
messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer briefly."},
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
```

Training for 1 epoch, batch_size=2, lr=1e-4, weight_decay=0.01 (L2 regularization coefficient, prevents overfitting), optim="paged_adamw_8bit" uses the 8-bit AdamW optimizer, which reduces memory usage and speeds up training.

For additional training, I used `Trainer`, an API for full-fledged training in PyTorch, and `TrainingArguments`, which offers a wide range of parameters for configuring model training.

Gradient accumulation technology was used to create the effect of a larger package. In addition, using gradient control points saved memory for intermediate activations.

The training process on the A100 GPU required about 11 GB of video memory (due to the small amount of training data).

### 2.3 Evaluation Metric
Accuracy (correct prediction of object counts):

$$ 𝐴𝑐𝑐𝑢𝑟𝑎𝑐𝑦 = \frac{\text{number of correct predictions}}{\text{total test samples}} \times 100 $$

Accuracy was calculated by comparing extracted numerical answers from model responses with ground truth labels using  `accuracy_score(y_true, y_pred)` from `sklearn.metrics`.

## 3. Test results

To test the model without training and after fine-tuning, the same test dataset was used (to assess the basic quality of the model and understand how effective the fine-tuning was).

All model predictions and correct answers are stored in files in the files folder.

### 3.1 Zero-Shot Setup

When testing a model without additional training, it was fed one instance of the dataset at a time.

Without additional training, the model on a random test sample of 998 objects gave the correct answer in 64.9% of cases (correctly predicted the answer to 648 objects out of 998). The model deviated from the correct answer by more than one. This may be due to the fact that some items were placed close to each other or were located behind each other, and the model could count them as one object.

Additionally, the model was tested on a slightly larger test dataset (1750 objects) (one of the first attempts). In that case, without additional training, the model on a random test sample of 1750 instances gave the correct answer in 48.4% of cases (correctly predicted the answer on 847 objects out of 1750). On the remaining objects, the model was wrong by one.

### 3.2 Supervised Fine-Tuning (SFT) Setup
Fine-tuning the model took place during one epoch on a small training dataset (3500 objects in total).

Two objects were fed to the model to get the prediction. Objects were converted using the additional function `collate_fn`, which returned data to be passed to the model input in the format that it needed. After receiving the prediction from the model, the numbers were extracted from the response and stored in a list.

After fine-tuning, the model achieved 100% accuracy, although L2 regularization was applied (i.e., protection against overfitting) and there was no data leakage (the model was trained on the training dataset, and tested on the test one). It is important to note that the input data was in the version that the model expects.

The code shows an example of testing a model when the input data was different from what the model expected. The accuracy in this case was 34.2%.
 
