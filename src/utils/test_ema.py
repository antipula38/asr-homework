import unittest

import torch
from torch import nn

from src.utils.ema import ModelEMA


class TestModelEMA(unittest.TestCase):

    def test_initialization(self):
        """EMA должна создавать независимую копию модели."""

        model = nn.Linear(1, 1)
        model.train()

        ema = ModelEMA(model, decay=0.9)

        # EMA и основная модель — разные объекты
        self.assertIsNot(ema.model, model)

        # Параметры первоначально совпадают
        for p, ema_p in zip(
            model.parameters(), ema.model.parameters()
        ):
            self.assertTrue(torch.equal(p, ema_p))

            # Но они не разделяют память
            self.assertNotEqual(
                p.data_ptr(), ema_p.data_ptr()
            )

            # EMA не должна вычислять градиенты
            self.assertFalse(ema_p.requires_grad)

        # EMA работает в режиме eval
        self.assertFalse(ema.model.training)

        # Исходная модель остаётся в train
        self.assertTrue(model.training)

    def test_parameter_update(self):
        """Проверка формулы EMA на двух последовательных шагах."""

        model = nn.Linear(1, 1, bias=False)

        with torch.no_grad():
            model.weight.fill_(2.0)

        ema = ModelEMA(model, decay=0.9)

        # Первый шаг: основной вес становится 4
        with torch.no_grad():
            model.weight.fill_(4.0)

        ema.update(model)

        # 0.9 * 2 + 0.1 * 4 = 2.2
        self.assertAlmostEqual(
            ema.model.weight.item(), 2.2, places=5
        )

        # Основная модель не изменилась
        self.assertAlmostEqual(
            model.weight.item(), 4.0, places=5
        )

        # Второй шаг: основной вес становится 6
        with torch.no_grad():
            model.weight.fill_(6.0)

        ema.update(model)

        # 0.9 * 2.2 + 0.1 * 6 = 2.58
        self.assertAlmostEqual(
            ema.model.weight.item(), 2.58, places=5
        )

        self.assertAlmostEqual(
            model.weight.item(), 6.0, places=5
        )

    def test_batchnorm_buffers(self):
        """Buffers BatchNorm должны копироваться в EMA."""

        model = nn.BatchNorm1d(2)
        ema = ModelEMA(model, decay=0.9)

        # Имитируем изменение статистик BatchNorm
        with torch.no_grad():
            model.running_mean.copy_(
                torch.tensor([0.5, 1.0])
            )
            model.running_var.copy_(
                torch.tensor([1.5, 2.0])
            )
            model.num_batches_tracked.fill_(7)

        ema.update(model)

        self.assertTrue(
            torch.equal(
                ema.model.running_mean,
                model.running_mean,
            )
        )

        self.assertTrue(
            torch.equal(
                ema.model.running_var,
                model.running_var,
            )
        )

        self.assertTrue(
            torch.equal(
                ema.model.num_batches_tracked,
                model.num_batches_tracked,
            )
        )

        # EMA остаётся в режиме eval
        self.assertFalse(ema.model.training)

    def test_invalid_decay(self):
        """Decay должен принадлежать интервалу [0, 1)."""

        model = nn.Linear(1, 1)

        with self.assertRaises(AssertionError):
            ModelEMA(model, decay=1.0)

        with self.assertRaises(AssertionError):
            ModelEMA(model, decay=-0.1)


if __name__ == "__main__":
    unittest.main()