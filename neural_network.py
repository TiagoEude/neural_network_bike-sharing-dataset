import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Seta os números de nós na entrada, nós na camada oculta e nós na camada de saída
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Inicia os pesos
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        # Seta a função de ativação como uma função sigmoid
        # Nota: em Python, e possivel definir uma função com uma expressão lambida
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))
                    

    def train(self, features, targets):
        ''' Treina a rede neural com um lote de registros e alvos
        
            Argumentos
            ---------
            
            features: Array 2D, cada linha é um registro de dados, cada coluna é uma característica
            targets: Array 1D com os alvos
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            # Chama a função de propagação direta
            final_outputs, hidden_outputs = self.forward_pass_train(X)  
            # Chama a função de retropropagação
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implementação da função de propagação direta ####
        
        # Camada oculta
        inputs = np.array(X, ndmin=2).T
        inputs = inputs.reshape((1, -1))
        hidden_inputs = np.dot(inputs, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Camada de saída
        final_inputs = hidden_outputs.dot(self.weights_hidden_to_output)
        final_outputs = final_inputs
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implementação da função de retropropagação ####
        new_X = np.array(X, ndmin=2)
        new_X = new_X.reshape((1, -1))
        new_y = np.array(y, ndmin=2)
        new_y = new_y.reshape((1, -1))

        # Calculo do erro
        error = final_outputs - new_y
        
        # Erro de retropopagação
        output_error_term =  error

        hidden_output_error_term = np.dot(output_error_term, self.weights_hidden_to_output.T)
        hidden_input_error_term = hidden_output_error_term * hidden_outputs * (1 - hidden_outputs)

        # Delta pesso (input to hidden)
        delta_weights_i_h += np.dot(new_X.T, hidden_input_error_term)
        # Delta pesso (hidden to output)
        delta_weights_h_o += np.dot(hidden_outputs.T, output_error_term)
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += -self.lr * delta_weights_h_o / n_records
        self.weights_input_to_hidden += -self.lr * delta_weights_i_h / n_records

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implementação da propagação direta ####
        # Camada oculta
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Camada de saída
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs 
        
        return final_outputs


#########################################################
# Seta os hiperparâmetros aqui
##########################################################
iterations = 3000
learning_rate = 0.1
hidden_nodes = 15
output_nodes = 1
