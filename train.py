import time
from options.train_options import TrainOptions
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from models.models import ModelsFactory
from utils.tb_visualizer import TBVisualizer
from collections import OrderedDict
import os


class Train:
    def __init__(self):
        self._opt = TrainOptions().parse()
        data_loader_train = CustomDatasetDataLoader(self._opt, is_for_train=True)
        data_loader_test = CustomDatasetDataLoader(self._opt, is_for_train=False)

        self._dataset_train = data_loader_train.load_data()
        self._dataset_test = data_loader_test.load_data()

        self._dataset_train_size = len(data_loader_train)
        self._dataset_test_size = len(data_loader_test)
        print('#train images = %d' % self._dataset_train_size)
        print('#test images = %d' % self._dataset_test_size)

        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._tb_visualizer = TBVisualizer(self._opt)

        self._train()

    def _train(self):
        self._total_steps = self._opt.load_epoch * self._dataset_train_size
        self._iters_per_epoch = self._dataset_train_size / self._opt.batch_size
        self._last_display_time = None
        self._last_save_latest_time = None
        self._last_print_time = time.time()

        for i_epoch in range(self._opt.load_epoch + 1, self._opt.nepochs_no_decay + self._opt.nepochs_decay + 1):
            epoch_start_time = time.time()

            # train epoch
            self._train_epoch(i_epoch)

            # save model
            print('saving the model at the end of epoch %d, iters %d' % (i_epoch, self._total_steps))
            self._model.save(i_epoch)

            # print epoch info
            time_epoch = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, self._opt.nepochs_no_decay + self._opt.nepochs_decay, time_epoch,
                   time_epoch / 60, time_epoch / 3600))

            # update learning rate
            if i_epoch > self._opt.nepochs_no_decay:
                self._model.update_learning_rate()

    def _train_epoch(self, i_epoch):
        epoch_iter = 0
        self._model.set_train()
        for i_train_batch, train_batch in enumerate(self._dataset_train):
            iter_start_time = time.time()

            # display flags
            do_visuals = self._last_display_time is None or time.time() - self._last_display_time > self._opt.display_freq_s
            do_print_terminal = time.time() - self._last_print_time > self._opt.print_freq_s or do_visuals

            # train model
            self._model.set_input(train_batch)
            train_generator = ((i_train_batch+1) % self._opt.train_G_every_n_iterations == 0) or do_visuals
            self._model.optimize_parameters(keep_data_for_visuals=do_visuals, train_generator=train_generator)

            # update epoch info
            self._total_steps += self._opt.batch_size
            epoch_iter += self._opt.batch_size

            # display terminal
            if do_print_terminal:
                self._display_terminal(iter_start_time, i_epoch, i_train_batch, do_visuals)
                self._last_print_time = time.time()

            # display visualizer
            if do_visuals:
                self._display_visualizer_train(self._total_steps)
                self._display_visualizer_val(i_epoch, self._total_steps)
                self._last_display_time = time.time()

            # save model
            if self._last_save_latest_time is None or time.time() - self._last_save_latest_time > self._opt.save_latest_freq_s:
                print('saving the latest model (epoch %d, total_steps %d)' % (i_epoch, self._total_steps))
                self._model.save(i_epoch)
                self._last_save_latest_time = time.time()

    def _display_terminal(self, iter_start_time, i_epoch, i_train_batch, visuals_flag):
        errors = self._model.get_current_errors()
        t = (time.time() - iter_start_time) / self._opt.batch_size
        self._tb_visualizer.print_current_train_errors(i_epoch, i_train_batch, self._iters_per_epoch, errors, t, visuals_flag)

    def _display_visualizer_train(self, total_steps):
        self._tb_visualizer.display_current_results(self._model.get_current_visuals(), total_steps, is_train=True)
        self._tb_visualizer.plot_scalars(self._model.get_current_errors(), total_steps, is_train=True)
        self._tb_visualizer.plot_scalars(self._model.get_current_scalars(), total_steps, is_train=True)

    def _display_visualizer_val(self, i_epoch, total_steps):
        val_start_time = time.time()

        # set model to eval
        self._model.set_eval()

        # evaluate self._opt.num_iters_validate epochs
        val_errors = OrderedDict()
        for i_val_batch, val_batch in enumerate(self._dataset_test):
            if i_val_batch == self._opt.num_iters_validate:
                break

            # evaluate model
            self._model.set_input(val_batch)
            self._model.forward(keep_data_for_visuals=(i_val_batch == 0))
            errors = self._model.get_current_errors()

            # store current batch errors
            for k, v in errors.iteritems():
                if k in val_errors:
                    val_errors[k] += v
                else:
                    val_errors[k] = v

        # normalize errors
        for k in val_errors.iterkeys():
            val_errors[k] /= self._opt.num_iters_validate

        # visualize
        t = (time.time() - val_start_time)
        self._tb_visualizer.print_current_validate_errors(i_epoch, val_errors, t)
        self._tb_visualizer.plot_scalars(val_errors, total_steps, is_train=False)
        self._tb_visualizer.display_current_results(self._model.get_current_visuals(), total_steps, is_train=False)

        # set model back to train
        self._model.set_train()


if __name__ == "__main__":
    Train()
