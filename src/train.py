from learner import get_learner

if __name__ == "__main__":
    learner = get_learner()
    learner.fit_one_cycle(1)

    learner.save("weights")
