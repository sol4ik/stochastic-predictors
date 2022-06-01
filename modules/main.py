from experiment.experiment import StochasticExperiment

from optparse import OptionParser


CONFIG_FILE = "../experiments/exp-b01/config-b01.yaml"


if __name__ == "__main__":
    op = OptionParser()
    op.add_option("-c", "--config", type=str, default=CONFIG_FILE,
                  help="relative path to experiment configuration file")

    (opts, args) = op.parse_args()
    o = dict(**vars(opts))

    experiment = StochasticExperiment(o["config"])
    experiment.set_experiment()
    experiment.run_training()
