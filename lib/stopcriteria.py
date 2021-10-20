import error
import numpy as np

class StopCriteria:
    def __init__(self, max_evaluations=None, max_iterations=None,
                 max_evals_woimp=None, max_iter_woimp=None,
                 cost_function_threshold=None, improvement_threshold=None):
        if (max_evaluations is None and max_iterations is None
                and max_evals_woimp is None and max_iter_woimp
                and cost_function_threshold is None):
            raise error.Error('StopCriteria.__init__: at least one criterium'
                              + ' must be given.')
        if ((max_evals_woimp is not None or max_iter_woimp is not None)
                and improvement_threshold is None):
            raise error.MissingInputError('StopCriteria.__init__',
                                          'improvement_threshold')
        self.max_evals = max_evaluations
        self.max_iter = max_iterations
        self.max_evals_woimp = max_evals_woimp
        self.max_iter_woimp = max_iter_woimp
        self.objfun_threshold = cost_function_threshold
        self.imp_threshold = improvement_threshold
        self.last_fx = None
        self.last_nevals = None
        self.last_niter = None
        self.mewi_counter = None
        self.miwi_counter = None

    def reset_memory(self):
        self.last_fx = 1e20
        self.last_nevals = 0
        self.last_niter = 0
        self.mewi_counter = 0
        self.miwi_counter = 0

    def stop(self, number_evaluations, number_iterations,
             current_best_evaluation):
        FLAG = False
        if self.max_evals is not None and number_evaluations >= self.max_evals:
            FLAG = True
        if self.max_iter is not None and number_iterations >= self.max_iter:
            FLAG = True
        if (self.objfun_threshold is not None
                and current_best_evaluation <= self.objfun_threshold):
            FLAG = True

        if self.max_evals_woimp is not None or self.max_iter_woimp is not None:
            if current_best_evaluation > self.last_fx:
                improvement = False
            elif (self.last_fx != 0
                    and np.abs(self.last_fx-current_best_evaluation)
                    / np.abs(self.last_fx)*100 <= self.imp_threshold):
                improvement = False
            elif (self.last_fx == 0
                    and self.last_fx == current_best_evaluation):
                improvement = False
            else:
                improvement = True

        if self.max_evals_woimp is not None:
            if not improvement:
                self.mewi_counter += number_evaluations-self.last_nevals
                if self.mewi_counter >= self.max_evals_woimp:
                    FLAG = True
            else:
                self.mewi_counter = 0

        if self.max_iter_woimp is not None:
            if not improvement:
                self.miwi_counter += number_iterations-self.last_niter
                if self.miwi_counter >= self.max_iter_woimp:
                    FLAG = True
            else:
                self.miwi_counter = 0
        self.last_fx = current_best_evaluation
        self.last_nevals = number_evaluations
        self.last_niter = number_iterations
        return FLAG

    def copy(self, new=None):
        if new is None:
            new = StopCriteria(self.max_evals, self.max_iter,
                               self.max_evals_woimp, self.max_iter_woimp,
                               self.objfun_threshold, self.imp_threshold)
            new.last_fx = self.last_fx
            new.last_nevals = self.last_nevals
            new.last_niter = self.last_niter
            new.mewi_counter = self.mewi_counter
            new.miwi_counter = self.miwi_counter
            return new
        else:
            self.max_evals = new.max_evals
            self.max_iter = new.max_iter
            self.max_evals_woimp = new.max_evals_woimp
            self.max_iter_woimp = new.max_iter_woimp
            self.objfun_threshold = new.objfun_threshold
            self.imp_threshold = new.imp_threshold
            self.last_fx = new.last_fx
            self.last_nevals = new.last_nevals
            self.last_niter = new.last_niter
            self.mewi_counter = new.mewi_counter
            self.miwi_counter = new.miwi_counter

    def __str__(self):
        message = 'Stop Criteria'
        if self.max_evals is not None:
            message += '\nMaximum number of evaluations: %d' % self.max_evals
        if self.max_evals_woimp is not None:
            message += ('\nMaximum number of evaluations without improvement: '
                        + '%d' % self.max_evals_woimp)
        if self.last_nevals is not None:
            message += ('\nLast current number of evaluations recorded: %d'
                        % self.last_nevals)
        if self.mewi_counter is not None:
            message += ('\nCurrent number of evaluations without improvement: '
                        + '%d' % self.mewi_counter)
        if self.max_iter is not None:
            message += '\nMaximum number of iterations: %d' % self.max_iter
        if self.max_iter_woimp is not None:
            message += ('\nMaximum number of iterations without improvement: '
                        + '%d' % self.max_iter_woimp)
        if self.last_niter is not None:
            message += ('\nLast current number of iterations recorded: %d'
                        % self.last_niter)
        if self.miwi_counter is not None:
            message += ('\nCurrent number of iterations without improvement: '
                        + '%d' % self.miwi_counter)
        if self.imp_threshold is not None:
            message += '\nImprovement threshold: %.1f%%' % self.imp_threshold
        if self.objfun_threshold is not None:
            message += ('\nObjective Function Threshold: %.2e'
                        % self.objfun_threshold)
        if self.last_fx is not None:
            message += '\nLast best fitness recorded: %.2e' % self.last_fx
        return message