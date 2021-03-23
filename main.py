# This is a sample Python script.
import MLDS
import numpy as np
import matplotlib.pyplot as plt

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    M = MLDS.Logistic(r=3.8)
    U = M.trajectory(.3, 500)

    net = MLDS.models.RCN(n_reservoir=50, transient=10
                          , read_out_type="augmented"
                          )

    Y_train = net.train(U, U + 1)
    net.predict(U)
    plt.plot(U.T)
    plt.show()

    print(U[:, 0])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
