from typing import Literal
import torch


def intersection_over_union(
    box_preds: torch.Tensor,
    box_labels: torch.Tensor,
    box_format: Literal["corners", "midpoints"] = "corners",
):
    if box_format == "corners":
        box1_x1, box2_x1 = box_preds[..., 0:1], box_labels[..., 0:1]
        box1_x2, box2_x2 = box_preds[..., 1:2], box_labels[..., 1:2]
        box1_y1, box2_y1 = box_preds[..., 2:3], box_labels[..., 2:3]
        box1_y2, box2_y2 = box_preds[..., 3:4], box_labels[..., 3:4]
    elif box_format == "midpoints":
        # TODO:
        pass

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    l1 = abs(box1_x2 - box1_x1)
    b1 = abs(box1_y2 - box1_y1)
    l2 = abs(box2_x2 - box2_x1)
    b2 = abs(box2_y2 - box2_y1)
    union = (l1 * b1) + (l2 * b2) - intersection + 1e-6

    return intersection / union


def non_max_suppression(box_preds, probablity, iou_threshold):
    pass


def assert_close(actual, expected, atol=1e-6):
    assert torch.allclose(actual, expected, atol=atol), (
        f"\nExpected: {expected}\nGot: {actual}"
    )


if __name__ == "__main__":
    # Test 1: Identical boxes
    pred = torch.tensor([[1.0, 3.0, 1.0, 3.0]])
    gt = torch.tensor([[1.0, 3.0, 1.0, 3.0]])
    assert_close(intersection_over_union(pred, gt), torch.tensor([[1.0]]))

    # Test 2: No overlap
    pred = torch.tensor([[1.0, 2.0, 1.0, 2.0]])
    gt = torch.tensor([[3.0, 4.0, 3.0, 4.0]])
    assert_close(intersection_over_union(pred, gt), torch.tensor([[0.0]]))

    # Test 3: Partial overlap
    pred = torch.tensor([[1.0, 4.0, 1.0, 4.0]])
    gt = torch.tensor([[2.0, 5.0, 2.0, 5.0]])

    assert_close(intersection_over_union(pred, gt), torch.tensor([[4 / 14]]))

    # Test 4: One box inside another
    pred = torch.tensor([[1.0, 5.0, 1.0, 5.0]])
    gt = torch.tensor([[2.0, 4.0, 2.0, 4.0]])
    assert_close(intersection_over_union(pred, gt), torch.tensor([[0.25]]))

    # Test 5: Touching edges
    pred = torch.tensor([[1.0, 2.0, 1.0, 2.0]])
    gt = torch.tensor([[2.0, 3.0, 1.0, 2.0]])
    assert_close(intersection_over_union(pred, gt), torch.tensor([[0.0]]))

    # Test 6: Symmetry
    # IoU(A,B) == IoU(B,A)
    pred = torch.tensor([[1.0, 4.0, 1.0, 4.0]])
    gt = torch.tensor([[2.0, 5.0, 2.0, 5.0]])

    iou1 = intersection_over_union(pred, gt)
    iou2 = intersection_over_union(gt, pred)

    assert_close(iou1, iou2)
