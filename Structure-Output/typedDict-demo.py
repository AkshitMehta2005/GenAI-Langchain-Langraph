from typing import TypedDict


class Person(TypedDict):
    name:str
    age:int
    


person1:Person = {'name':"akshit","age":21}


print(person1)