from abc import ABC, abstractmethod

# 1. Define the interface (The contract)
class Runnable(ABC):
    @abstractmethod
    def invoke(self, input_data):
        pass

# 2. Implement some simple runnables
class Doubler(Runnable):
    def invoke(self, input_data):
        return input_data * 2

class Adder(Runnable):
    def invoke(self, input_data):
        return input_data + 10

# 3. Create a chain runner that executes them sequentially
class SimpleChain(Runnable):
    def __init__(self, runnables):
        self.runnables = runnables

    def invoke(self, input_data):
        current_val = input_data
        for runnable in self.runnables:
            current_val = runnable.invoke(current_val)
        return current_val

# 4. Use the chain
doubler = Doubler()
adder = Adder()

# Chain: (Input * 2) + 10
my_chain = SimpleChain([doubler, adder])

result = my_chain.invoke(5)
print(f"Result of chain: {result}")  # Expected: (5 * 2) + 10 = 20