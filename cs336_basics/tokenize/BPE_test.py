import pickle


path = '/home/yatin/Documents/Wolverine/Class/CS336LLM/CS336-assignment1-basics/tests/_snapshots/test_train_bpe_special_tokens.pkl'

with open(path, "rb") as f:
    expected_data = pickle.load(f)
print("expected data:", list(expected_data['vocab_values'])[255])

print("len of vocab:", len(expected_data['vocab_values']))
print("len of vocab keys:", len(expected_data['vocab_keys']))
print("len of merges:", len(expected_data['merges']))