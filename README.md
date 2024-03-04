### [Text2Decision Pipeline Overview](https://openreview.net/forum?id=fEoemPDicz&referrer=%5Bthe%20profile%20of%20Hua-Dong%20Xiong%5D(%2Fprofile%3Fid%3D~Hua-Dong_Xiong1)

The **Text2Decision** pipeline is an approach to pre-train a neural network model with the goal of mapping semantic space to decision-related cognitive space.

#### Components

- **Preprocess.py**
  - Generates training prompts.
  - Embeds questions for neural network understanding.
  - Defines decision embeddings.

- **Data_loader.py**
  - Patches data for training, testing, and validation phases.
  - Performs data normalizations.

- **model.py**
  - Contains the neural network model for training.
  - Allows modification and addition of model classes for comparison.

- **inference.py**
  - Conducts inference on think-aloud data.
  - Analyzes predictions, individual differences, context-dependence, and visualization.

- **util.py**
  - Provides miscellaneous functions for ML, visualization, and more.
