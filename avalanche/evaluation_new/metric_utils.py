import io

import matplotlib.pyplot as plt
from PIL import Image
from numpy import ndarray
from sklearn.metrics import ConfusionMatrixDisplay


def default_cm_image_creator(confusion_matrix_tensor: ndarray,
                             display_labels=None,
                             include_values=True,
                             xticks_rotation='horizontal',
                             values_format=None,
                             cmap='viridis',
                             dpi=50,
                             image_title=''):
    """
    The default Confusion Matrix image creator. This utility uses Scikit-learn
    `ConfusionMatrixDisplay` to create the matplotlib figure. The figure
    is then converted to a PIL `Image`.

    For more info about the accepted graphic parameters, see:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html#sklearn.metrics.plot_confusion_matrix.

    :param confusion_matrix_tensor: The tensor describing the confusion matrix.
        This can be easily obtained through Scikit-learn `confusion_matrix`
        utility.
    :param display_labels: Target names used for plotting. By default, `labels`
        will be used if it is defined, otherwise the values will be inferred by
        the matrix tensor.
    :param include_values: Includes values in confusion matrix. Defaults to
        `True`.
    :param xticks_rotation: Rotation of xtick labels. Valid values are
        'vertical', 'horizontal' or a float point value. Defaults to
        'horizontal'.
    :param values_format: Format specification for values in confusion matrix.
        Defaults to `None`, which means that the format specification is
        'd' or '.2g', whichever is shorter.
    :param cmap: Must be a str or a Colormap recognized by matplotlib.
        Defaults to 'viridis'.
    :param dpi: The dpi to use to save the image.
    :param image_title: The title of the image. Defaults to an empty string.
    :return: The Confusion Matrix as a PIL Image.
    """

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix_tensor, display_labels=display_labels)
    display.plot(include_values=include_values, cmap=cmap,
                 xticks_rotation=xticks_rotation, values_format=values_format)

    display.ax_.set_title(image_title)

    fig = display.figure_
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='jpg', dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)
    return image


__all__ = ['default_cm_image_creator']
