import matplotlib.pyplot as plt

class Liquid:
    def __init__(self, error):
        self.current_error = error
        self.last_action = 0

    def take_action(self, action):
        self.current_error += 0.1 * action 
        self.last_action = action

class PID:
    err_sum = 0
    old_err = 0

    def pid(self, current, goal, kp, ki, kd):
        err = goal - current
        self.err_sum += err
        delta_err = err - self.old_err
        self.old_err = err

        return kp*err + ki*self.err_sum + kd*delta_err

class Derivative: 
    def __init__(self):
        self.last_x = 0
        self.last_y = 0

    def get_gradient(self, x, y):
        d = (y - self.last_y) / (x - self.last_x)
        self.last_x = x
        self.last_y = y

        return d

class Train: 
    kp = 1.0; ki = 0.5; kd = 0.5
    goal = 0
    episode_length = 100
    learning_rate = 0.001 

    def __init__(self):
        self.dp = Derivative()
        self.di = Derivative()
        self.dd = Derivative()
        self.step = 0
        self.last_loss = 0

    def abs_mean(self, list):
        sum = 0
        for i in list:
            sum += abs(i)
        return sum / len(list)

    def loss(self):
        liquid = Liquid(10) 
        pid = PID()

        error = []
        for i in range(self.episode_length): 
            error.append(liquid.current_error)
            liquid.take_action(pid.pid(liquid.current_error, self.goal, self.kp, self.ki, self.kd))

        return self.abs_mean(error)

    def optimize(self):
        self.kp = self.kp - self.learning_rate * self.dp.get_gradient(self.kp, self.loss())
        self.ki = self.ki - self.learning_rate * self.di.get_gradient(self.ki, self.loss())
        self.kd = self.kd - self.learning_rate * self.dd.get_gradient(self.kd, self.loss())
        self.last_loss = self.dd.last_y
        print("step={}, kp={}, ki={}, kd={}, loss={}".format(self.step, self.kp, self.ki, self.kd, self.last_loss))
        self.step += 1

if __name__ == '__main__': 
    loss = []
    kp = []
    ki = []
    kd = []

    train = Train() 

    for j in range(100000):
        train.optimize()
        loss.append(train.last_loss)  
        kp.append(train.kp)
        ki.append(train.ki)
        kd.append(train.kd)

    plt.plot(loss, label="loss")
    plt.plot(kp, label="Kp")
    plt.plot(ki, label="Ki")
    plt.plot(kd, label="Kd")
    plt.legend()
    plt.title("Train")
    plt.savefig("train.png")

    