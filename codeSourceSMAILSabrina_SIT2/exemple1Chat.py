import numpy as np
import pandas as p
import random
from random import randint

# on spécifie les parametres utiles pour l'apprentissage
episodes = 3000  # un episode est une experience complète de l'agent qui s'achéve soit en gagnant ou en perdant
# taux d'apprentissage ou le learning rate est le degré d'acceptation de la nouvelle valeur par rapport à l'ancienne

taux_apprentissage = 0.4
gamma = 0.95  # facteur d'actualisation, equilibre la récompense immédiate et future
epsilon = 0.2  # taux d'exploration


# ***********************************************************************
# dans ce cas pratique on n'utilise pas la librairie gym, on ne possède
# pas d'environnement donc on doit le créer
# on crée donc une matrice de 4 lignes, 4 colonnes qui correspond à l'exemple
# ou on met les récompenses et les pénalités dans les cellules
# on initialise aussi le point de départ du chat


class EnvChat(object):
    def __init__(self) -> None:
        super(EnvChat, self).__init__()

        self.grille = np.array([[0, 0, 1, 0],
                                [0, -2, 0, 0],
                                [0, 0, 0, -1],
                                [0, 0, 0, 0]])
        # using pandas
        p.DataFrame(self.grille)
        # les positions de debut
        self.ligne = 3
        self.colonne = 0

        # les actions possibles
        # i ligne, j colonne
        self.actions = [
            [-1, 0],  # Up un pas en haut i--
            [1, 0],  # Down un pas en bas i++
            [0, -1],  # Left pas a gauche j--
            [0, 1]  # Right pas à droite j++
        ]

    def reset(self):

        self.ligne = 3
        self.colonne = 0
        return(self.ligne*4, self.colonne+1)

    def step(self, action):

        self.ligne = max(0, min(self.ligne + self.actions[action][0], 3))
        self.colonne = max(0, min(self.colonne + self.actions[action][1], 3))
        return (self.ligne*4 + self.colonne+1), self.grille[self.ligne][self.colonne]

    def show(self):
        """
            Show the grid
        """
        print("----------la grille-----------")
        i = 0
        for line in self.grille:
            j = 0
            for pt in line:
                print("%s\t" %
                      (pt if j != self.colonne or i != self.ligne else "CHAT"), end="")
                j += 1
            i += 1
            print("")

    def is_finished(self):
        if (self.grille[self.ligne][self.colonne] == 1):
            return True
        else:
            return False


# afficher la grille
env = EnvChat()
env.show()

# initialiser la qtable à 0


def initialiserQ(nbetats, nbactions):
    return np.zeros((nbetats, nbactions))


qtable = initialiserQ(17, 4)
p.DataFrame(qtable)

for episode in range(episodes):
    # reset l'environnement, retourne à l'etat intial
    etat = env.reset()
    while not env.is_finished():

        if random.uniform(0, 1) < epsilon:  # exploration
            action = randint(0, 3)

        else:  # exploitation
            action = np.argmax(qtable[etat])

        # prendre les informations grace à step
        nvetat, reward = env.step(action)
        # update la q table
        qtable[etat, action] = qtable[etat, action] + taux_apprentissage * \
            (reward + gamma *
             qtable[nvetat][np.argmax(qtable[nvetat])] - qtable[etat, action])

        # letat
        etat = nvetat

print('-------------La qtable :--------------')
print(qtable[1:17])
