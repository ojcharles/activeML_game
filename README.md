lets lay out what this is:

so this is a app that will let benchmark a few different active ML approaches.
it will also allow a user to play against various active ML algorithms. i.e. can i beat it?


what this means in terms of code:
so i will need a dataset, preferably a smallish one so its fast and not overwhelming.
test train split.
Test:
  This is used to assess the predictive performance of models trained on currently labelled data
train:
  this provides some initial seed data, and is the space that users and AML can select from to label.

The process will look like a set of batches, say multiple rounds of chosing 5 things.
After each batch, a model is trained and metrics plotted.