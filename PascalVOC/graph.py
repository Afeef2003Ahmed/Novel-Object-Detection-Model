from CustomModel_VOC.Dataset import load_checkpoint as chkpt_voc
from CustomModel_Self_DrivingCars.Dataset import load_checkpoint as chkpt_sdc
from CustomModel_BDD100K.Dataset import load_checkpoint as chkpt_bdd100k
import matplotlib.pyplot as plt

loaded_checkpoint_VOC = chkpt_voc()


if chkpt_voc is not None:
    train_loss_values_voc = chkpt_voc["epoch_losses"]

epochs_voc = [i for i in range(len(train_loss_values_voc))]



plt.plot(epochs_voc,train_loss_values_voc,marker='o', linestyle='-', color='b', label='VOC Dataset')

plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')

plt.show()

plt.savefig('/raid/cs21resch15003/Afeef_Intern/CutsomModel_VOC/loss_curve_merged')
