import io, os, torch

import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from mplplots import plots


class Reporter:
    def __init__(self, n4m):
        self.n4m = n4m

    def exportReport(self, report_path):
        c = canvas.Canvas(report_path, pagesize=letter)
        width, height = letter

        for key, value in self.n4m.model_def['Minimizers'].items():
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            plots.plot_training(ax, f"Training Loss of {key}", key, self.n4m.train_losses[key], self.n4m.val_losses[key])
            training = io.BytesIO()
            plt.savefig(training, format='png')
            training.seek(0)
            plt.close()
            c.drawString(100, height - 30, f"Training Loss of {key}")
            c.drawImage(ImageReader(training), 50, height - 290, width=500, height=250)
            c.showPage()

        for key in self.n4m.model_def['Minimizers'].keys():
            c.drawString(100, height - 30, f"Prediction of {key}")
            for ind, name_data in enumerate(self.n4m.prediction.keys()):
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(111)
                plots.plot_results(ax, name_data, key, self.n4m.prediction[name_data][key]['A'],
                               self.n4m.prediction[name_data][key]['B'], self.n4m.model_def["SampleTime"])
                # Add a text box with correlation coefficient
                results = io.BytesIO()
                plt.savefig(results, format='png')
                results.seek(0)
                plt.close()
                c.drawImage(ImageReader(results), 50, height - 290 - 245*ind, width=500, height=250)
            c.showPage()
        c.save()
