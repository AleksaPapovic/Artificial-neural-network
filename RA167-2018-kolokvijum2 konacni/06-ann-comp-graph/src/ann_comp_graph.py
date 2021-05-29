from abc import abstractmethod
import math
import copy
import pandas as pd
from matplotlib import pyplot
import random
from tqdm import trange 
from numpy.random import RandomState
from sklearn import preprocessing
from imblearn.over_sampling import ADASYN,SMOTE


random.seed(1337)


class ComputationalNode(object):

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dz):
        pass


class MultiplyNode(ComputationalNode):

    def __init__(self):
        self.x = [0., 0.]  # x[0] je ulaz, x[1] je tezina

    def forward(self, x):
        self.x = x
        # TODO 1: implementirati forward-pass za mnozac
        return self.x[0] * self.x[1]

    def backward(self, dz):
        # TODO 1: implementirati backward-pass za mnozac
        return [dz * self.x[1], dz * self.x[0]]





class SumNode(ComputationalNode):

    def __init__(self):
        self.x = []  # x je vektor, odnosno niz skalara

    def forward(self, x):
        self.x = x
        # TODO 2: implementirati forward-pass za sabirac
        return sum(self.x)

    def backward(self, dz):
        # TODO 2: implementirati backward-pass za sabirac
        return [dz for xx in self.x]





class SigmoidNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x je skalar

    def forward(self, x):
        self.x = x
        return self._sigmoid(self.x)

    def backward(self, dz):
        # TODO 3: implementirati backward-pass za sigmoidalni cvor
        return dz * self._sigmoid(self.x) * (1. - self._sigmoid(self.x))

    def _sigmoid(self, x):
        # TODO 3: implementirati sigmoidalnu funkcij
        try:
            return 1. / (1. + math.exp(-x))
        except:
            return float('inf')





class ReluNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x is an input

    def forward(self, x):
        self.x = x
        return self._relu(self.x)

    def backward(self, dz):
        return dz * (1. if self.x > 0. else 0.)

    def _relu(self, x):
        return max(0., x)


class LinNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x is an input

    def forward(self, x):
        self.x = x
        return self._lin(self.x)

    def backward(self, dz):
        return dz

    def _lin(self, x):
        return x


class TanhNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x is an i

    def forward(self, x):
        self.x = x
        return self._tanh(self.x)

    def backward(self, dz):
        return dz * (1 - self._tanh(self.x) ** 2)

    def _tanh(self, x):
        try:
            return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        except:
            return float('inf')


class NeuronNode(ComputationalNode):

    def __init__(self, n_inputs, activation):
        self.n_inputs = n_inputs  # moramo da znamo kolika ima ulaza da bismo znali koliko nam treba mnozaca
        self.multiply_nodes = []  # lista mnozaca
        self.sum_node = SumNode()  # sabirac

        # TODO 4: napraviti n_inputs mnozaca u listi mnozaca, odnosno mnozac za svaki ulaz i njemu odgovaraj
        # za svaki mnozac inicijalizovati tezinu na broj iz normalne (gauss) raspodele sa st. devijacijom 0.
        for i in range(self.n_inputs):
            mn = MultiplyNode()
            mn.x = [1., random.gauss(0., 0.1)]
            self.multiply_nodes.append(mn)

        # TODO 5: dodati jos jedan mnozac u listi mnozaca, za bias
        # bias ulaz je uvek fiksiran na 1.
        # bias tezinu inicijalizovati na broj iz normalne (gauss) raspodele sa st. devijacijom 0.01
        mn = MultiplyNode()  # init bias node
        mn.x = [1., random.gauss(0., 0.01)]  # init bias weight
        self.multiply_nodes.append(mn)

        # TODO 6: ako ulazni parametar funckije 'activation' ima vrednosti 'sigmoid',
        # inicijalizovati da aktivaciona funckija bude sigmoidalni cvor
        if activation == 'sigmoid':
            self.activation_node = SigmoidNode()
        elif activation == 'relu':
            self.activation_node = ReluNode()
        elif activation == 'lin':
            self.activation_node = LinNode()
        elif activation == 'tanh':
            self.activation_node = TanhNode()
        else:
            raise RuntimeError('Unknown activation function "{0}".'.format(activation))

        self.previous_deltas = [0.] * (self.n_inputs + 1)
        self.gradients = []

    def forward(self, x):  # x je vektor ulaza u neuron, odnosno lista skalara
        x = copy.copy(x)
        x.append(1.)  # uvek implicitino dodajemo bias=1. kao ulaz

        # TODO 7: implementirati forward-pass za vestacki neuron
        # u x se nalaze ulazi i bias neurona
        # iskoristi forward-pass za mnozace, sabirac i aktivacionu funkciju da bi se dobio konacni izlaz iz
        for_sum = []
        for i, xx in enumerate(x):
            inp = [x[i], self.multiply_nodes[i].x[1]]
            for_sum.append(self.multiply_nodes[i].forward(inp))
        summed = self.sum_node.forward(for_sum)
        summed_act = self.activation_node.forward(summed)
        return summed_act

    def backward(self, dz):
        dw = []
        # print(dz)
        d = dz[0] if type(dz[0]) == float else sum(dz)  # u d se nalazi spoljasnji gradijent izlaza neurona

        # TODO 8: implementirati backward-pass za vestacki neuron
        # iskoristiti backward-pass za aktivacionu funkciju, sabirac i mnozace da bi se dobili gradijenti te
        # izracunate gradijente tezina ubaciti u listu dw
        act = self.activation_node.backward(d)
        summed_act = self.sum_node.backward(act)
        for i, bb in enumerate(summed_act):
            dw.append(self.multiply_nodes[i].backward(bb)[1])

        self.gradients.append(dw)
        return dw

    def update_weights(self, learning_rate, momentum):
        # azuriranje tezina vestackog neurona
        # learning_rate je korak gradijenta

        # TODO 11: azurirati tezine neurona (odnosno azurirati drugi parametar svih mnozaca u neuronu)
        # gradijenti tezina se nalaze u list self.gradients
        for i, multiply_node in enumerate(self.multiply_nodes):
            mean_grad = sum([grad[i] for grad in self.gradients]) / len(self.gradients)
            delta = learning_rate * mean_grad + momentum * self.previous_deltas[i]
            self.multiply_nodes[i].x[1] -= delta
            self.previous_deltas[i] = delta

        self.gradients = []  # ciscenje liste gradijenata (da sve bude cisto za sledecu iteraciju)


class NeuralLayer(ComputationalNode):

    def __init__(self, n_inputs, n_neurons, activation):
        self.n_inputs = n_inputs  # broj ulaza u ovaj sloj neurona
        self.n_neurons = n_neurons  # broj neurona u sloju (toliko ce biti i izlaza iz ovog sloja)
        self.activation = activation  # aktivaciona funkcija neurona u ovom sloju

        self.neurons = []
        # konstruisanje sloja nuerona
        for _ in range(n_neurons):
            neuron = NeuronNode(n_inputs, activation)
            self.neurons.append(neuron)

    def forward(self, x):  # x je vektor, odnosno lista "n_inputs" elemenata
        layer_output = []
        # forward-pass za sloj neurona je zapravo forward-pass za svaki neuron u sloju nad zadatim ulazom x
        for neuron in self.neurons:
            neuron_output = neuron.forward(x)
            layer_output.append(neuron_output)

        return layer_output

    def backward(self, dz):  # dz je vektor, odnosno lista "n_neurons" elemenata
        dd = []
        # backward-pass za sloj neurona je zapravo backward-pass za svaki neuron u sloju nad
        # zadatim spoljasnjim gradijentima dz
        for i, neuron in enumerate(self.neurons):
            neuron_dz = [d[i] for d in dz]
            neuron_dz = neuron.backward(neuron_dz)
            dd.append(neuron_dz[:-1])  # izuzimamo gradijent za bias jer se on ne propagira unazad

        return dd

    def update_weights(self, learning_rate, momentum):
        # azuriranje tezina slojeva neurona je azuriranje tezina svakog neurona u tom sloju
        for neuron in self.neurons:
            neuron.update_weights(learning_rate, momentum)


class NeuralNetwork(ComputationalNode):

    def __init__(self):
        self.layers = []  # neuronska mreza se sastoji od slojeva neurona

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):  # x je vektor koji predstavlja ulaz u neuronsku mrezu
        # TODO 9: implementirati forward-pass za celu neuronsku mrezu
        # ulaz za prvi sloj neurona je x
        # ulaz za sve ostale slojeve izlaz iz prethodnog sloja
        prev_layer_output = None
        for idx, layer in enumerate(self.layers):
            if idx == 0:  # input layer
                prev_layer_output = layer.forward(x)
            else:
                prev_layer_output = layer.forward(prev_layer_output)
        return prev_layer_output  # actually an output from last layer

    def backward(self, dz):
        # TODO 10: implementirati forward-pass za celu neuronsku mrezu
        # spoljasnji gradijent za izlazni sloj neurona je dz
        # spoljasnji gradijenti za ostale slojeve su izracunati gradijenti iz sledeceg sloja
        next_layer_dz = None
        for idx, layer in enumerate(self.layers[::-1]):
            if idx == 0:
                next_layer_dz = layer.backward(dz)
            else:
                next_layer_dz = layer.backward(next_layer_dz)
        return next_layer_dz

    def update_weights(self, learning_rate, momentum):
        # azuriranje tezina neuronske mreze je azuriranje tezina slojeva
        for layer in self.layers:
            layer.update_weights(learning_rate, momentum)

    def fit(self, X, Y, learning_rate=0.1, momentum=0.0, nb_epochs=10, shuffle=False, verbose=0):
        assert len(X) == len(Y)

        hist = []  # za plotovanje funkcije greske kroz epohe
        for epoch in trange(nb_epochs):
            if shuffle:  # izmesati podatke
                random.seed(epoch)
                random.shuffle(X)
                random.seed(epoch)
                random.shuffle(Y)

            total_loss = 0.0
            for x, y in zip(X, Y):
                y_pred = self.forward(x)  # forward-pass da izracunamo izlaz
                y_target = y  # zeljeni izlaz
                grad = 0.0
                for p, t in zip(y_pred, y_target):
                    total_loss += 0.5 * (t - p) ** 2.  # funkcija greske je kvadratna greska
                    # total_loss +=self.CrossEntropy(p, t)
                    grad += -(t - p)  # gradijent funkcije greske u odnosu na izlaz
                    # print(grad)
                # backward-pass da izracunamo gradijente tezina
                self.backward([[grad]])
                # azuriranje tezina na osnovu izracunatih gradijenata i koraka "learning_rate"
                self.update_weights(learning_rate, momentum)

            if verbose == 1:
                print('Epoch {0}: loss {1}'.format(epoch + 1, total_loss))
            hist.append(total_loss)

        print('Loss: {0}'.format(total_loss))
        return hist

    def predict(self, x):
        return self.forward(x)


def encodinghot(fajl, kolone):
    ulaz = fajl
    for kolona in kolone:
        ulaz  = pd.concat([ulaz, pd.get_dummies(ulaz[kolona], prefix=kolona, dummy_na=False)], axis=1).drop([kolona], axis=1)
    return  ulaz

def normalizovanje(dataframe):
    vrednosti = dataframe.values
    min_max_skaler = preprocessing.MinMaxScaler()
    vrednosti_skalirane = min_max_skaler.fit_transform(vrednosti)
    return pd.DataFrame(vrednosti_skalirane)

if __name__ == '__main__':

    # pod a)
    smokes = smoked = never_smoked = 0
    male = female = other = 0
    maried = not_married = 0
    private = selfemployed = govt = children = 0
    pyplot.style.use('ggplot')
    vrednosti = pd.read_csv('../data/dataset.csv').values
    for red in vrednosti:

        if red[11] == 1:
            if red[10] == 'smokes':
                smokes += 1
            elif red[10] == 'formerly smoked':
                smoked += 1
            elif red[10] == 'never smoked':
                never_smoked += 1
    for red in vrednosti:
        if red[11] == 1:
            if red[1] == 'Male':
                male += 1
            elif red[1] == 'Female':
                female += 1
            elif red[1] == 'Other':
                other += 1
    for red in vrednosti:
        if red[11] == 1:
            if red[5] == 'Yes':
                maried += 1
            elif red[5] == 'No':
                not_married += 1
    for red in vrednosti:
        if red[11] == 1:
            if red[6] == 'children':
                children += 1
            elif red[6] == 'Govt_job':
                govt += 1
            elif red[6] == 'Self-employed':
                selfemployed += 1
            elif red[6] == 'Private':
                private += 1

    types_smokers = ['Smoker', 'Non Smoker', 'Former Smoker']
    smokers_strokes = [smokes, never_smoked, smoked]

    types_smokers_pos = [i for i, _ in enumerate(types_smokers)]

    pyplot.bar(types_smokers_pos, smokers_strokes, color='green')
    pyplot.xlabel("Types of smokers")
    pyplot.ylabel("Strokes")
    pyplot.title("Strokes in correlation to smokers")

    pyplot.xticks(types_smokers_pos, types_smokers)
    pyplot.show()

    genders = ['Male', 'Female', 'Other']
    genders_strokes = [male, female, other]
    genders_pos = [i for i, _ in enumerate(genders)]
    pyplot.bar(genders_pos, genders_strokes, color='green')
    pyplot.xlabel("Genders")
    pyplot.ylabel("Strokes")
    pyplot.title("Strokes in correlation to genders")
    pyplot.xticks(genders_pos, genders)
    pyplot.show()

    marrieds = ['Married', 'Not married']
    marrieds_strokes = [maried, not_married]
    married_pos = [i for i, _ in enumerate(marrieds)]
    pyplot.bar( marrieds,  marrieds_strokes, color='green')
    pyplot.xlabel("Marrieds")
    pyplot.ylabel("Strokes")
    pyplot.title("Strokes in correlation to marrieds")
    pyplot.xticks( married_pos, marrieds)
    pyplot.show()

    posao = ['Private', 'Self-employed', 'Govt_job','chlidren']
    posao_strokes = [private, selfemployed, govt ,children]
    posao_pos = [i for i, _ in enumerate(posao)]
    pyplot.bar(posao_pos, posao_strokes, color='green')
    pyplot.xlabel("Works")
    pyplot.ylabel("Strokes")
    pyplot.title("Strokes in correlation to work")
    pyplot.xticks(posao_pos, posao)
    pyplot.show()


    #B deo
    nn = NeuralNetwork()

    col_list = ["gender", "age","hypertension","heart_disease","ever_married",
                "work_type","Residence_type","avg_glucose_level","bmi","smoking_status","stroke"]


    papo_ulaz  = pd.read_csv('../data/dataset.csv',usecols=col_list)
    papo_izlaz = pd.read_csv('../data/dataset.csv', usecols=["stroke"])

    papo_ulaz.fillna(0, inplace=True)
    papo_izlaz.fillna(0, inplace=True)


    hotovane = ["gender", "ever_married", "Residence_type", "work_type", "smoking_status"]


    papo_ulaz = encodinghot(papo_ulaz, hotovane)
    papo_ulaz.to_csv(path_or_buf='../data/papo_ulaz.csv', index=False)


    ada = ADASYN()
    nova_lista = [member for member in range(22) if member != 5]
    X_resampled, y_resampled = ada.fit_resample(papo_ulaz.iloc[:, nova_lista], papo_ulaz['stroke'])

    # oversample = SMOTE()
    # X_resampled, y_resampled= oversample.fit_resample(papo_ulaz.iloc[:, nova_lista], papo_ulaz['stroke']

    papo_ulaz = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)


    papo_izlaz = papo_ulaz["stroke"]

    papo_ulaz_0=  papo_ulaz[papo_ulaz["stroke"] == 0]
    papo_ulaz_1=  papo_ulaz[papo_ulaz["stroke"] == 1]

    papo_ulaz_0 = papo_ulaz_0.drop(["stroke"], axis=1)
    papo_ulaz_1 = papo_ulaz_1.drop(["stroke"], axis=1)


    papo_ulaz_0.to_csv(path_or_buf='../data/papo_ulaz_0.csv', index=False)
    papo_ulaz_1.to_csv(path_or_buf='../data/papo_ulaz_1.csv', index=False)

    papo_ulaz_0 = normalizovanje(papo_ulaz_0)
    papo_ulaz_1 = normalizovanje(papo_ulaz_1)

    #oversampling :D
    #duzina_ulaza_1 = len(papo_ulaz_1)
    #duzina_ulaza_0 = len(papo_ulaz_0)

    #papo_ulaz_1_frames = [papo_ulaz_1,papo_ulaz_1,papo_ulaz_1,papo_ulaz_1,papo_ulaz_1,
    #                      papo_ulaz_1,papo_ulaz_1,papo_ulaz_1,papo_ulaz_1,papo_ulaz_1,papo_ulaz_1]
    #papo_ulaz_1 = pd.concat(papo_ulaz_1_frames)

    papo_ulaz_0.to_csv(path_or_buf='../data/ulaz_0.csv', index=False)
    papo_ulaz_1.to_csv(path_or_buf='../data/ulaz_1.csv', index=False)
    papo_izlaz.to_csv(path_or_buf='../data/izlaz.csv', index=False)

    nn.add(NeuralLayer(21, 21, 'tanh'))
    nn.add(NeuralLayer(21, 10, 'tanh'))
    nn.add(NeuralLayer(10, 1, 'sigmoid'))

    #polovina =  int((duzina_ulaza_1 + duzina_ulaza_0)/2)


    #papo_ulaz_0 = papo_ulaz_0[:polovina]
    #papo_ulaz_1 = papo_ulaz_1[:polovina]


    ptrain_ulaz_1_7 =  papo_ulaz_1.sample(frac=0.35, random_state=RandomState())
    ptrain_ulaz_1_3 =  papo_ulaz_1.loc[~papo_ulaz_1.index.isin(ptrain_ulaz_1_7.index)]

    ptrain_ulaz_0_7 = papo_ulaz_0.sample(frac=0.35, random_state=RandomState())
    ptrain_ulaz_0_3 = papo_ulaz_0.loc[~papo_ulaz_0.index.isin(ptrain_ulaz_0_7.index)]


    ptrain_ulaz_frames = [ptrain_ulaz_1_7,ptrain_ulaz_0_7]
    ptest_ulaz_frames = [ptrain_ulaz_1_3, ptrain_ulaz_0_3]

    #ptrain_ulaz = papo_ulaz.sample(frac=0.35, random_state=RandomState())
    #ptest_ulaz  = papo_ulaz.loc[~papo_ulaz.index.isin(ptrain_ulaz.index)]

    ptrain_ulaz = pd.concat(ptrain_ulaz_frames)
    ptest_ulaz = pd.concat(ptest_ulaz_frames)

    # racnomerna raspodela izlaz
    izlaz_1 = []
    izlaz_0 = []

    for row in papo_izlaz.values.tolist():
        if row == 1:
            izlaz_1.append(row)
        elif row == 0:
            izlaz_0.append(row)
    df_izlaz1 = pd.DataFrame(izlaz_1)
    df_izlaz0 = pd.DataFrame(izlaz_0)
    ptrain_izlaz_1_7_1 = df_izlaz1.sample(frac=0.35, random_state=RandomState())
    ptrain_izlaz_1_3_1 = df_izlaz1.loc[~df_izlaz1.index.isin(ptrain_izlaz_1_7_1.index)]

    ptrain_izlaz_1_7_0 = df_izlaz0.sample(frac=0.35, random_state=RandomState())
    ptrain_izlaz_1_3_0 = df_izlaz0.loc[~df_izlaz0.index.isin(ptrain_izlaz_1_7_0.index)]

    ptrain_izlaz_frames = [ptrain_izlaz_1_7_1, ptrain_izlaz_1_7_0]
    ptest_izlaz_frames = [ptrain_izlaz_1_3_1, ptrain_izlaz_1_3_0]

    ptrain_izlaz = pd.concat(ptrain_izlaz_frames)
    ptest_izlaz = pd.concat(ptest_izlaz_frames)
    #ptrain_izlaz[1:]
    ptrain_ulaz.to_csv(path_or_buf='../data/trainingulaz.csv', index=False)
    ptrain_izlaz.to_csv(path_or_buf='../data/trainingizlaz.csv', index=False)


    X= ptrain_ulaz.values.tolist()
    Y= ptrain_izlaz.values.tolist()

    if input("Treniraj mrezu?(y/n)") == 'y':
         # plotovanje funkcije greske
        history = nn.fit( X, Y, learning_rate=0.01, momentum=0.9, nb_epochs=10, shuffle=True, verbose=1)
        pyplot.plot(history)

        tp = tn = fp = fn = 0
        precision = []
        test_X = ptest_ulaz.values.tolist()
        test_Y = ptest_izlaz.values.tolist()
        #matrica konfuzije
        for i, j in zip(test_X, test_Y):
            print([j[0],nn.predict(i)[0]])
            if j[0] == 1:
                if nn.predict(i)[0] > 0.5:
                    tp += 1
                elif nn.predict(i)[0] < 0.5:
                    fn += 1
            elif j[0] == 0:
                if nn.predict(i)[0] < 0.5:
                    tn += 1
                elif nn.predict(i)[0] > 0.5:
                    fp += 1

            # if j[0] == 1 and 0.0 < j[0] - nn.predict(i)[0] < 0.5:
            #    fp += 1
            # elif j[0] == 0 and -0.5 < j[0] - nn.predict(i)[0] < 0.0:
            #     tn += 1
            # elif j[0] == 1 and 0.5 < j[0] - nn.predict(i)[0] < 1.0:
            #     tp += 1
            # elif j[0] == 0 and -1 < j[0] - nn.predict(i)[0] < -0.5:
            #     fn += 1
        print( tp)
        print(tn)
        print( fp)
        print( fn)
        try:
            accuracy = (tp + tn) / (tp + fp + fn + tn)
        except:
            accuracy = 0
        try:
            precision = tp / (tp + fp)
        except:
            precision = 0
        try:
            recall = tp / (tp + fn)
        except:
            recall = 0
        try:
            f1 = (precision * recall) / (precision + recall)
        except:
            f1 = 0

        print('Accuracy', round(accuracy * 100, 2), '%')
        print('Precision:', round(precision, 2))
        print('Recall:', round(recall, 2))
        print('F1 je', round(2*f1, 2))
        pyplot.show()