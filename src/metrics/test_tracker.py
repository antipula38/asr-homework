from src.metrics.tracker import MetricTracker


tracker = MetricTracker("WER")

tracker.update_ratio("WER", 30, 100)
tracker.update_ratio("WER", 20, 200)

expected = 50 / 300
actual = tracker.avg("WER")

assert abs(actual - expected) < 1e-9, (actual, expected)

print("basic ratio test passed")


tracker = MetricTracker("WER")

tracker.update_ratio("WER", 5, 0)
tracker.update_ratio("WER", 20, 100)

expected = 25 / 100
actual = tracker.avg("WER")

assert abs(actual - expected) < 1e-9, (actual, expected)

print("empty target test passed")