import numpy
import matplotlib.pyplot as plot

if __name__ == '__main__':
    data = numpy.loadtxt('../cnn_history0.2D.csv', skiprows=1, delimiter=',')
    epoch    = data[:,0].flatten()
    loss     = data[:,1].flatten()
    acc     = data[:,2].flatten()
    # lr       = data[:,3].flatten()
    val_loss  = data[:,3].flatten()
    val_acc = data[:,4].flatten()
    
    fig, axis1 = plot.subplots()
    axis1.plot(epoch, acc, 'b-', label='acc', zorder=1)
    axis1.plot(epoch, val_acc, 'r-', label='val_acc', zorder=1)
    axis1.set_xlabel('epoch')
    axis1.set_ylabel('accuracy')
    handles, labels = axis1.get_legend_handles_labels()
    axis1.legend(handles, labels)

    axis2 = axis1.twinx()
    axis2.plot(epoch, loss, 'b-', label='loss', zorder=1)
    axis2.plot(epoch, val_loss, 'r-', label='val_loss', zorder=1)
    axis2.set_ylabel('loss')
    handles, labels = axis2.get_legend_handles_labels()
    axis2.legend(handles, labels)

    # lr_min = float(numpy.min(lr))
    # lr_max = float(numpy.max(lr))
    # for i in range(epoch.shape[0]):
    #     ratio = None
    #     if (lr_max - lr_min) == 0:
    #         ratio = 0.0
    #     else:
    #         ratio = 1 - (float(lr[i]) - lr_min) / (lr_max - lr_min)
    #     plot.axvspan(i - 0.5, i + 0.5,
    #                  facecolor=str(ratio),
    #                  alpha=0.5,
    #                  zorder=0)
    
    fig.tight_layout()
    plot.show()
    # plot.savefig('trace.svg', format='svg', dpi=1200)

