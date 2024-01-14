# {<[(BRACKETS)]>}

The purpose of this toy project is to investigate how different model architectures deal with a simple classification task. We will be building classification models to predict whether a sequence of bracket symbols is "gramatically" correct. 

### Rules

We define the grammar in the following way:
- All bracket symbols are grouped into pairs, with one opening and one closing, eg: ```()```, ```[]```, ```{}```, ```<>```.
- Every bracket opened should be closed. No closing bracket can exist if its corresponding opening hasn't come earlier. Eg ```(())```, ```[()]```, ```({})``` are correct; ```)(```, ```{}}```, ```{{{}``` are not.
- Brackets are opened and closed in a last in-first out manner, ie any closing bracket must correspond to the most recent yet unclosed opening one. Eg ```[{({})}]()``` is correct, ```[(])``` isn't.
- The padding elements (```.```) at the end of a sequence are ignored, and the input is corect if the brackets are correct. Eg ```{}[]....``` is correct because it is the same as ```{}[]```; ```{[]}))...``` is not because ```{[]}))``` isn't
- Sequences in which the padding symbol comes before or inbetween any brackets are considered incorrect. Eg ```{[(...)]}...```, ```[]..()...``` are incorrect even if the brackets themselves are OK. A sequence consisting only of padding and no brackets (```......```) is also considered incorrect.

### The elephant in the room

The task described above can be easily achieved with explicit programming: a simple algorithm can very quickly tell us whether a sequence is correct with 100% accuracy and without the need for training an ML model. However, the simplicity of the task has some advantages. Firstly, we can generate and label with 100% accuracy a large datatset. Secondly, the nature of the classification task implies that all bias is avoidable - that is, training a model to be 100% accurate is not inherently impossible. There might also be some benefit in terms of attempts to explain our models' "thinking".

That said, I should state that the purposes of this project are purely pedagogical and exploratory.

### APPROACH

We will be building a series of ML model with increasing complexity and comparing their performance. We will start with "vanilla" neural networks (NNs), comparing width vs depth, then move on to convolutional NNs, and finally try out more elaborate sequence models such as GRUs, LSTMs, and transformers.

We will be using PyTorch for handling data and buidling/training the models.
