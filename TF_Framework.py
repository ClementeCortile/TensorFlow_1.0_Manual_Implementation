
import numpy as np

#Recreating Tensorflow framework

class Operation():
    """
    no docstring
    """
    #Class Attributes
    #WARNING None

    #Defining Constructor
    def __init__(self, input_nodes=[]):
        #object Operation has two nodes
        self.input_nodes = input_nodes
        self.output_nodes = []

        #every operation is appended to the output node
        for node in input_nodes:
            node.output_nodes.append(self)

        #global var declared in Graph Class
        _default_graph.operation.append(self)

    #Compute method will be overwritten by the operations' classes
    def compute(self):
        pass


#Add inheriths methods from Operation
class add(Operation):

    #Add takes in x and y and adds them
    def __init__(self, x, y):
        #super call to receive arguments from the inherithed class
        super().__init__([x,y])


    #Overwriting Compute method from Operation Class
    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var


class multiply(Operation):

    def __init__(self, x, y):
        super().__init__([x,y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var * y_var


class matmul(Operation):

    def __init__(self, x, y):
        super().__init__([x,y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var.dot(y_var)

#######################################################


#Graph class will connect placeholder, variables and operations
class Graph():

    def __init__(self):
        self.operation = []
        self.placeholders = []
        self.variables = []

    def set_as_default(self):
        global _default_graph
        _default_graph = self

#Placeholder class will hold the values of the nodes
class Placeholder():

    def __init__(self):

        self.output_nodes = []
        _default_graph.placeholders.append(self)

#Changeable parameters of the graph ( the weights)
class Variable():

    def __init__(self, initial_value=None):

        self.value = initial_value
        self.output_nodes = []

        _default_graph.variables.append(self)


#This function will define the order of the operations (tree traversal order)
def traverse_postorder(operation):
    """
    PortOrder Traversal of Nodes. Basically makes sure computations
    are done in the correct order (Ax first, then Ax + b).
    """

    nodes_postorder = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder

#Provides flow control for the operations
class Session():

    #feed_dict maps placeholders to input values
    # later they'll feed batches of data to the network
    def run(self, operation, feed_dict={}):

        nodes_postorder = traverse_postorder(operation)

        for node in nodes_postorder:

            #IF NOT A Placeholder
            if type(node) == Placeholder:
                node.output = feed_dict[node]
            #And NOT an operation
            elif type(node) == Variable:
                node.output = node.value
            #Then EXECUTE the operation for all the nodes
            else:
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs) # * to handle multiple args

            if type(node.output) == list:
                #converting the list to numpy arrays
                node.output = np.array(node.output)

        return operation.output

#Declaring session object
sess = Session()

"""
Building a line using the TF_framework

z = Ax + b
A = 10
b = 1
z = 10x + 1
X is a placeholder
"""

g = Graph()
g.set_as_default()
#Naming operations as z = Ax + b, x = 10 passed in the dictionary
A = 10
b = 1

x = Placeholder()

y = multiply(A,x)
z = add(y,b)

result = sess.run(operation=z, feed_dict={x:10})

print(result)

sess = Session()

#Declaring a graph object for a matrix multiplication
g = Graph()
#setting default graph
g.set_as_default()
#Passing the variables
A = Variable([[10,20],[30,40]])
b = Variable([1,1])
#Allocating the placeholder
x = Placeholder()
#Assigning the value of the operation to y
y = matmul(A,x)



sess.run(operation = z, feed_dict={x:10})
