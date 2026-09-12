import os
import time
from typing import Dict, Any, Optional

try:
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def export_to_onnx(
    model_name_or_path: str = "l3cube-pune/hing-roberta",
    output_path: str = "models/hinglish_sentiment.onnx"
) -> bool:
    """
    Exports a HuggingFace PyTorch transformer checkpoint to ONNX format.
    Requires torch and transformers.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Loading checkpoint {model_name_or_path} for ONNX export...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        model.eval()

        dummy_input = tokenizer(
            "bhai yeh badhiya hai",
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )

        print(f"Exporting ONNX model to {output_path}...")
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size"}
            },
            opset_version=14
        )
        print(f"ONNX export succeeded: {output_path}")
        return True
    except Exception as e:
        print(f"ONNX export encountered: {e}")
        return False


def quantize_onnx_int8(
    input_onnx_path: str = "models/hinglish_sentiment.onnx",
    quantized_path: str = "models/hinglish_sentiment_int8.onnx"
) -> Optional[Dict[str, Any]]:
    """
    Applies INT8 Dynamic Quantization on the ONNX graph using ONNX Runtime.
    Reduces model size by ~75% and speeds up CPU inference by 3x-4x.
    """
    if not ONNX_AVAILABLE:
        print("onnxruntime is not installed. Please install onnxruntime.")
        return None

    if not os.path.exists(input_onnx_path):
        print(f"Input ONNX file {input_onnx_path} does not exist.")
        return None

    os.makedirs(os.path.dirname(quantized_path), exist_ok=True)
    orig_size_mb = os.path.getsize(input_onnx_path) / (1024 * 1024)

    print(f"Applying INT8 Dynamic Quantization: {input_onnx_path} -> {quantized_path}")
    quantize_dynamic(
        model_input=input_onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QInt8
    )

    quant_size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
    compression_ratio = round((1.0 - (quant_size_mb / orig_size_mb)) * 100, 2)

    return {
        "original_size_mb": round(orig_size_mb, 2),
        "quantized_size_mb": round(quant_size_mb, 2),
        "compression_percentage": compression_ratio,
        "quantized_path": quantized_path
    }
