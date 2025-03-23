

class IndexEncoder:

    def __init__(self, collection):

        self.collection = set(collection)
        self.length = len(self.collection)
        self.idx_to_obj = list(self.collection)
        self.obj_to_idx = {obj:i for i, obj in enumerate(self.idx_to_obj)}

    def encode(self, x):
        return self.obj_to_idx[x]
    
    def decode(self, x):
        return self.idx_to_obj[x]
    
def generate(model, encoder, seed=None, num_tokens=100, separator= " "):
    if seed == None:
        seed = 0

    output = encoder.decode(seed)
    curr_token = seed
    
    for _ in range(num_tokens):
        curr_token = model.predict(curr_token)
        output += separator + encoder.decode(curr_token)

    return output