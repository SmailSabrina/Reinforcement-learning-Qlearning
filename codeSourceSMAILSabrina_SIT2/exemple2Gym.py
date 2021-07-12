import numpy as np
import gym
import random

# l'environnement choisi est frozenLake offert par la bibliothèque gym
# c'est un environnement à base de grille ou l'agent commence à partir
# d'un point S et doit atteindre son objectif qui est le point G ou se trouve le frisbee
# en passant par la surface gelée F qui est sans danger tout en evitant
# les trous : les points H


env = gym.make("FrozenLake-v0")

# on spécifie les parametres utiles pour l'apprentissage

episodes = 15000  # un episode est une experience complète de l'agent qui s'achéve soit en gagnant ou en perdant
# taux d'apprentissage ou le learning rate est le degré d'acceptation de la nouvelle valeur par rapport à l'ancienne
taux_apprentissage = 0.4
gamma = 0.95  # facteur d'actualisation, equilibre la récompense immédiate et future
nb_pasMax = 99  # nombre de pas maximum par episode

# les parametres d'exploration
epsilon = 0.2  # taux d'exploration


# on doit créer la table Q (etat, action) et l'initialiser à 0
# pour l'environnement offert par gym l'etat de la table q correspond à une observation
# et l'action de la table Q correspond à une action effectuée

etat = env.observation_space.n
action = env.action_space.n

# pour intialiser la table on utilise la fonction offerte par la bibliothèque
# numpy pour la mettre à zero

qtable = np.zeros((etat, action))


# jusqu'a la fin de l'apprentissage ou la fin des episodes
for episode in range(episodes):
    # reset l'environnement, le remettre à la case de départ
    etat = env.reset()
    pas = 0
    fini = False

    for pas in range(nb_pasMax):

        if random.uniform(0, 1) < epsilon:  # exploration
            action = env.action_space.sample()

        else:  # exploitation
            action = np.argmax(qtable[etat])

        # prendre les informations grace à step
        nvetat, reward, fini, info = env.step(action)
        # mise à jour de la q table
        # selon l'instruction suivante
        # Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action])

        qtable[etat, action] = qtable[etat, action] + taux_apprentissage * \
            (reward + gamma * np.max(qtable[nvetat]) - qtable[etat, action])

        # letat
        etat = nvetat
        if fini == True:
            break

print(qtable)


env.close()
