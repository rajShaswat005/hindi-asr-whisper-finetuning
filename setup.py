from setuptools import setup, find_packages

setup(
    name="hindi-asr-whisper-finetuning",
    version="0.1.0",
    description="End-to-end Hindi ASR pipeline using OpenAI Whisper-Small",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "evaluate>=0.4.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "pydub>=0.25.1",
        "jiwer>=3.0.0",
        "requests>=2.31.0",
        "tenacity>=8.2.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "python-multipart>=0.0.6",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "httpx>=0.25.0",
        ]
    },
)
