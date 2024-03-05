### Text2Decision Pipeline Overview

The **Text2Decision** pipeline is an approach to pre-train a neural network model with the goal of mapping semantic space to decision-related cognitive space. Please see the paper [here](https://openreview.net/forum?id=fEoemPDicz&referrer=%5Bthe%20profile%20of%20Hua-Dong%20Xiong%5D(%2Fprofile%3Fid%3D~Hua-Dong_Xiong1)).



#### Scripts

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
 
  #### Data
  - **c13k_problems.json**
    - The question dataset from [Peterson, et.al.,2021](https://www.science.org/doi/full/10.1126/science.abe2629)
    - We preprocess the data by formatting question prompts to inquire the text embeddings;
    - We also preprocess the data by defining their decision embeddings.
      
  - **musked_simmulated_behavioral_text_embeddings.npy**
    - The simulated dataset by GPT-4 doing think-aloud in the same risky decision-making task.
      #### Dataset Columns

      - `index`
      - `sub_id`
      - `choices`
      - `p1`
      - `v1`
      - `p2`
      - `v2`
      - `trial_id`
      - `think_aloud`
      - `word_count`
      - `question_prompt`
      - `persona`
      - `text_embeddings`(from index 12 to the rightmost)

    
