#!/bin/bash

echo "=================================="
echo "F1 Race ML Pipeline - Setup Script"
echo "=================================="
echo ""

# Check Python version
echo "🐍 Checking Python..."
python --version

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements_ml.txt --quiet

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ Error installing dependencies"
    exit 1
fi

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Launch Jupyter: jupyter notebook f1_race_ml_notebook.ipynb"
echo "2. Run the notebook (Kernel → Restart & Run All)"
echo "3. Review visualizations and model performance"
echo ""
echo "Expected runtime: 5-10 minutes"
echo "📊 Generated files will be saved automatically"
echo ""
