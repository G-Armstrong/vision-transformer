# Polymathic AI Assignment

You are working with a new collaborator who studies the dynamics of fictitious gases. Several weeks back, their lab discovered an experimental procedure that 
they believe leads to a new class of fictitious gas behavior. They believe this class may be common in nature, but that it has never been recorded since the 
difference with more typical gas behavior is extremely subtle. Nonetheless, they believe if this can be verified, it will be a major step forward in their field
and the entire fictitious gas community is very excited. 

Your collaborator would like to explore this possibility with machine learning. If they could take measurements of the gas state and classify whether or 
not the gas is in the new state or not, then they could evaluate their hypothesis in the wild. To do this, they have spent the previous 10 days running experiments
but with their new procedure and without it to generate train, validation, and test sets for their models. 

However, their model is not very accurate. To be comfortable using their model, **they would like to hit 83% accuracy on the test set**, but they have struggled
passing 65% so far with a simple CNN that ChatGPT wrote for them. Their colleagues in the Computer Science department have told them that *transformers* might
be more effective, but as fictitious gas experts, they are not familiar with the model.

They've now reached out to you for help. They given you the training code they've used to reach 65% accuracy which includes their model, basic training/eval loops,
and dataset. The data has been provided as HDF5 files containing 28x28 (MNIST-sized) uniformly spaced grids of measurements. Your collaborator is running their
code on either their lab CPU-based server or their personal gaming GPU so while you can use whatever you like for development, keep in mind that the goal
is not to train a large model.

Instructions:
- Create a github repository you'll share with us.
- Implement a vision transformer style model that can be used on the given data. There are no constraints on the specific model size or architecture family apart from the use of multi-headed attention somewhere in the model.
- Benchmark this model on the provided data. 
- Tune the model and code to reach the desired accuracy of 83% on the training set.
- Write a helper function that writes your test set predictions out to `outputs/predictions.txt` in the same style as the example. This is what will be used to evaluate the accuracy.
- Commit and push your fork containing the final code with the correct predictions.txt and provide your contact with a link to the completed repository. 
- Please do not spend more than a couple of hours on this. While the problem has been set up to allow much higher accuracy, there are no additional points given for doing so. This is intended to evaluate your ability to work independently within an existing codebase.
- Note: keep in mind what you know about your collaborator's background in deciding how to proceed. You may make any modifications necessary to achieve the goal, but these modifications should be appropriately documented to 
allow your collaborator to understand what changed. Please include a note to the collaborator explaining what you changed and why to reach your score. This is not a formal report - bullet points are fine if they are sufficiently clear.
