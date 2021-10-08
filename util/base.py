from abc import ABC


class AgentBase(ABC):
    def __init__(self, env):
        pass

    def _preprocess(self, observations, frozen=False):
        """
        Implement preprocess of the observation. that is, filtering, normalization etc.
        :param observations:
        :param frozen: if True, stop learning process such as update of filters
        :return:
        """
        raise NotImplementedError

    def sample_action(self, observation, frozen=False):
        """
        Implement the single sampled action
        :param observation:
        :param frozen: if True, stop learning process such as update of filters
        :return action, info:
        """
        raise NotImplementedError

    def sample_actions(self, observations, frozen=False):
        """
        Implement the multiple sampled action
        :param observations:
        :param frozen: if True, stop learning process such as update of filters
        :return actions, info:
        """
        raise NotImplementedError

    def _postprocess(self, actions, frozen=False):
        """
        Postprocess of the actions. such as scaling
        :param actions:
        :param frozen:
        :return processed_results:
        """
        raise NotImplementedError

    def update(self, observation, action, reward, next_observation, done, frozen=False):
        """
        Total update processes of the agent. This should include update of replay buffer, preprocess
        :param observation:
        :param action:
        :param reward:
        :param next_observation:
        :param done:
        :param frozen: if True, stop learning process
        :return:
        """
        raise NotImplementedError
