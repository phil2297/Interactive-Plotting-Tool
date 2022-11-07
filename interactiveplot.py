import os
import tkinter as tk
import webbrowser

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lmfit.models as mdl
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from sklearn.preprocessing import normalize

matplotlib.use('TkAgg')


class App(tk.Frame):
    # Essentials
    def __init__(self, master):
        """Initializing the frame in which things are created onto in
        the window construct."""
        tk.Frame.__init__(self, master)
        self.master = master
        self.master.title('Absorption/Emission line fitter')

    def dummy(self):
        """Dummy function to make non-implemented menu buttons
        do something instead of nothing."""

        window = tk.Toplevel(self)
        window.title('DUMMY')
        window.geometry('300x30')
        lbl_dummy = tk.Label(master=window, text='BUTTON NOT IMPLEMENTED')
        lbl_dummy.pack()

    def menu_bar(self):
        """Function that creates the menubar containing a file menu,
        an edit menu, a fitting menu, a formatting menu and a help menu."""

        # Initializing the menu-bar.
        menubar = tk.Menu(self.master, relief=tk.RAISED)
        self.master.config(menu=menubar)

        # Creating the drop down menu categories.
        filemenu = tk.Menu(menubar, tearoff=False)
        editmenu = tk.Menu(menubar, tearoff=False)
        fitmenu = tk.Menu(menubar, tearoff=False)
        formatmenu = tk.Menu(menubar, tearoff=False)
        helpmenu = tk.Menu(menubar, tearoff=False)

        # Creating the options in the "file" drop down menu.
        filemenu.add_command(label='Open File', command=self.open_file)
        filemenu.add_command(label='Open Files...', command=self.dummy)
        filemenu.add_command(label='Save plot', command=self.save_plot)
        filemenu.add_command(label='Save plot as...', command=self.save_plot_as)
        filemenu.add_separator()
        filemenu.add_command(label='Exit', command=self.master.destroy)

        # Creating the options in the "edit" drop down menu.
        editmenu.add_command(label='Zoom', command=self.zoom)
        editmenu.add_command(label='Smooth', command=self.smooth_data)
        editmenu.add_command(label='Cut', command=self.cut_data)
        editmenu.add_separator()
        normalizemenu = tk.Menu(editmenu, tearoff=False)
        normalizemenu.add_command(label='Normalize full data', command=self.normalize_data)
        normalizemenu.add_command(label='Normalize section...', command=self.normalize_section)
        editmenu.add_cascade(label='Normalize', menu=normalizemenu)

        # Creating the options in the "fit" drop down menu.
        basefuncs = tk.Menu(fitmenu, tearoff=False)
        periodicfuncs = tk.Menu(fitmenu, tearoff=False)
        peaklike_funcs = tk.Menu(fitmenu, tearoff=False)
        basefuncs.add_command(label='Linear', command=self.choose_linear)
        basefuncs.add_command(label='Polynomial', command=self.choose_polynomial)
        basefuncs.add_command(label='Exponential', command=self.choose_exponential)
        basefuncs.add_command(label='Power Law', command=self.choose_powerlaw)
        periodicfuncs.add_command(label='Sine', command=self.choose_sine)
        peaklike_funcs.add_command(label='Gaussian', command=self.choose_gauss)
        peaklike_funcs.add_command(label='Exponential Gauss', command=self.choose_exponentialgauss)
        peaklike_funcs.add_command(label='Lorentzian', command=self.choose_lorentzian)
        peaklike_funcs.add_command(label='Harmonic Oscillator', command=self.choose_harmonicoscillator)
        peaklike_funcs.add_command(label='Lognormal', command=self.choose_lognormal)
        fitmenu.add_cascade(label='Basic Functions', menu=basefuncs)
        fitmenu.add_cascade(label='Periodic Functions', menu=periodicfuncs)
        fitmenu.add_cascade(label='Peak-like Functions', menu=peaklike_funcs)
        fitmenu.add_separator()
        fitmenu.add_command(label='Save fit report as...', command=lambda: self.save_fit_report(self.result))
        # Creating the options in the "format" drop down menu.
        self.gridvar = tk.BooleanVar()
        formatmenu.add_checkbutton(label='Grid', variable=self.gridvar, 
                                   onvalue=True, offvalue=False, command=self.grid)
        formatmenu.add_command(label='Tight Layout', command=self.tight_layout)
        formatmenu.add_separator()
        formatmenu.add_command(label='X limits', command=self.x_lims)
        formatmenu.add_command(label='Y limits', command=self.y_lims)
        formatmenu.add_command(label='Labels', command=self.label_making)

        # Creating the options for the "help" drop down menu.
        helpmenu.add_command(label='Controls', command=self.dummy)
        helpmenu.add_command(label='Documentation', command=self.open_documentation)
        helpmenu.add_command(label='Contact', command=self.contact_info)

        # Initializing the drop down menus into the menu-bar.
        menubar.add_cascade(label='File', menu=filemenu)
        menubar.add_cascade(label='Edit', menu=editmenu)
        menubar.add_cascade(label='Fit', menu=fitmenu)
        menubar.add_cascade(label='Format', menu=formatmenu)
        menubar.add_cascade(label='Help?', menu=helpmenu)

    def draw_figure(self, master):
        """Function that creates a figure based on
        opened file."""
        try:
            self.figure_canvas.get_tk_widget().destroy()
            self.toolbar.destroy()
        except:
            pass
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.figure_canvas = FigureCanvasTkAgg(self.figure, master=master)
        self.figure_canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.figure_canvas, master, pack_toolbar=False)
        self.toolbar.update()

        self.axes = self.figure.add_subplot()
        self.axes.step(self.wavelength, self.flux, label='Data', where='mid')    
        self.axes.legend()    

        self.figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH,
                                                expand=True)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.BOTH)

    def reset(self, event):
        print('Resetting...')
        self.gridvar.set(False)
        self.figure_canvas.get_tk_widget().destroy()
        self.toolbar.destroy()
        self.fitwindow.destroy()
        self.vlinelist = []
        self.points = []
        self.wavelength = np.array(self.df['wavelength']).copy()
        self.flux = np.array(self.df['flux']).copy()*1e19
        self.draw_figure(master=self.master)
    

    # FILE MENU
    def open_file(self):
        """Function that loads/opens file by opening file
        explorer, then calls the draw_figure function to
        plot the data."""
        try:
            self.fitwindow.destroy()
            self.gridvar.set(False)
        except:
            pass
        self.points = []
        self.vlinelist = []
        self.filename = tk.filedialog.askopenfilename(initialdir=os.listdir(),
                                                      title = 'select a file',
                                                      filetypes = (('Data Files', '*.dat'),
                                                                   ('All Files', '*.*')))
        def create_data():
            self.scalingfactor_x = float(scalefactor_ents['ent_0'].get())
            self.scalingfactor_y = float(scalefactor_ents['ent_1'].get())
            scalewindow.destroy()
            scalewindow.update()
            self.df = pd.read_table(self.filename, delim_whitespace=True, header=None)
            self.NAME = os.path.basename(os.path.normpath(os.path.splitext(self.filename)[0]))
            self.df.name = self.NAME
            if len(self.df.columns) == 3:
                self.df.columns = ['wavelength', 'flux', 'error']
            else:
                self.df.columns = ['wavelength', 'flux', 'sum_flux', 'sky', 'error']
            self.wavelength = np.array(self.df['wavelength']).copy()*self.scalingfactor_x
            self.flux = np.array(self.df['flux']).copy()*self.scalingfactor_y
            self.draw_figure(master=self.master)
            
    
        scale_window = self.pop_up_window('Choose scaling factors', '250x75', create_data,
                                    label1='X scaling factor:', label2='Y scaling factor')
        scalewindow, scalefactor_ents = scale_window
    def open_files(self):
        pass

    def save_plot(self):
        self.figure.savefig(f'{self.NAME}.png', dpi=300)

    def save_plot_as(self):
        save_filename = tk.filedialog.asksaveasfilename(initialfile='Untitled.png',
                                                        defaultextension='.png',
                                                        filetypes=[('All Files', '*.*'),
                                                                   ('test','*.png')])
        self.figure.savefig(save_filename, dpi=300)

    def save_fit_report(self, current_fit_result):
        fit_report_filename = tk.filedialog.asksaveasfilename(initialfile='fit_report.txt',
                                                              defaultextension='.txt',
                                                              filetypes=[('All Files', '*.*')])
        with open(fit_report_filename, 'w') as fr:
            fr.write(current_fit_result.fit_report())

    # EDIT MENU
    def zoom(self):
        self.figure_canvas.mpl_connect('button_release_event', self.toolbar.zoom)

    def smooth_data(self):
        self.initialize_fit_frame()
        def smooth():
            box_pts = int(smooth_ent[f'ent_{0}'].get())
            box = np.ones(box_pts)/box_pts
            self.flux = np.convolve(self.flux, box, mode='same')
            self.draw_figure(master=self.fitwindow)
            self.fitwindow.focus_set()
            self.flux = np.array(self.df['flux'])

            smoothwindow.destroy()
            smoothwindow.update()
    
        smooth_window = self.pop_up_window('Smoothing level', '250x50',
                                           smooth, label1='Enter box smoothing')
        smoothwindow, smooth_ent = smooth_window
        
    def cut_data(self):
        self.initialize_fit_frame()
        def cut(event):
            self.fitregion()
            self.flux = np.delete(self.flux, self.goodrange)
            self.wavelength = np.delete(self.wavelength, self.goodrange)
            self.line_removal()
            self.draw_figure(master=self.fitwindow)
            self.fitwindow.focus_set()   
        self.fitwindow.bind('<Return>', cut)

    def normalize_data(self):
        self.initialize_fit_frame()
        self.flux = normalize(self.flux.reshape(1, -1))[0]
        self.draw_figure(master=self.fitwindow)
        self.flux = np.array(self.df['flux'])
        self.fitwindow.focus_set()
    
    def normalize_section(self):
        self.initialize_fit_frame()
        def norm(event):
            self.fitregion()
            self.flux[self.goodrange] = normalize(self.flux[self.goodrange].reshape(1, -1))[0]
            self.line_removal()
            self.draw_figure(master=self.fitwindow)
            self.flux = np.array(self.df['flux']).copy()
            self.fitwindow.focus_set()    
        self.fitwindow.bind('<Return>', norm)

    # FIT MENU
        # Required operational functions
    def draw_vline(self, event):
        """Draws a vertical line where the mouse button is located."""
        x = event.xdata
        y = event.ydata
        L =  self.axes.axvline(x=x, color='k', linestyle='dashed')
        self.vlinelist.append(L)
        self.points.append([x, y])
        self.figure_canvas.draw()

    def line_removal(self):
        for i in range(len(self.vlinelist)): 
            self.vlinelist.pop(0).remove()
            self.lines = np.delete(self.lines, 0)
            self.points.pop(0)

    def set_xlims(self, event):
        if len(self.points) > 1 and len(self.points) <= 2:
            xlimmin, xlimmax = self.points[0][0], self.points[1][0]
            self.axes.set_xlim(xlimmin-0.01*xlimmin,
                               xlimmax+0.01*xlimmax)
            print(xlimmin, xlimmax)
        self.figure_canvas.draw()

    def fitregion(self):
        self.lines = np.array(self.points)
        if self.lines[0][0] < self.lines[1][0]:
            self.rangemin = self.lines[0][0]
            self.rangemax = self.lines[1][0]
        elif self.lines[0][0] > self.lines[1][0]:
            self.rangemin = self.lines[1][0]
            self.rangemax = self.lines[0][0]
        self.goodregion = np.array([self.rangemin, self.rangemax])
        self.goodrange = np.where((self.wavelength >= self.goodregion[0]) &
                                  (self.wavelength <= self.goodregion[1]))

    def initialize_fit_frame(self):
        try:
            self.fitwindow.destroy()
            self.fitwindow.focus_set()
        except:
            pass
        self.gridvar.set(False)
        self.fitwindow = tk.Frame(self.master)
        self.draw_figure(master=self.fitwindow)
        self.clickvline = self.figure_canvas.mpl_connect('button_press_event', self.draw_vline)

        self.fitwindow.focus_set()
        self.fitwindow.bind('<space>', self.set_xlims)
        self.fitwindow.bind('<Escape>', self.reset)
       # self.pack_forget()
        self.fitwindow.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Functions to map to buttons choosing fit type
    def choose_linear(self):
        self.initialize_fit_frame()
        self.fitwindow.bind('<Return>', self.linear_fit)

    def choose_polynomial(self):
        self.initialize_fit_frame() 
        def update_degree():
            self.degree = int(polyent[f'ent_{0}'].get())
            poly_window.destroy()
            poly_window.update()
            
        poly_window, polyent = self.pop_up_window('Choose number of variables', 
                                                  '350x60', update_degree,
                                                  label1='Number of variables to fit (integer in [0,7]):')
        self.fitwindow.bind('<Return>', self.polynomial_fit)

    def choose_exponential(self):
        self.initialize_fit_frame()
        self.fitwindow.bind('<Return>', self.exponential_fit)
    
    def choose_powerlaw(self):
        self.initialize_fit_frame()
        self.fitwindow.bind('<Return>', self.powerlaw_fit)
    
    def choose_sine(self):
        self.initialize_fit_frame()
        self.fitwindow.bind('<Return>', self.sine_fit)

    def choose_gauss(self):
        self.initialize_fit_frame()
        self.fitwindow.bind('<Return>', self.gaussian_fit)
    
    def choose_exponentialgauss(self):
        self.dummy()
        
    def choose_lorentzian(self):
        self.dummy()
        
    def choose_harmonicoscillator(self):
        self.dummy()
        
    def choose_lognormal(self):
        self.dummy()
    
    
    def plot_fit(self):
        self.line_removal()
        self.axes.legend()
        self.figure_canvas.draw()
        self.vlinelist = []
        self.points = []
        
        # Functions that creates and fits the different models
    def linear_fit(self, event):
        """Fits a simple linear line to a chosen region."""
        self.fitregion()
        model = mdl.LinearModel()
        params = model.guess(self.flux[self.goodrange],
                             x=self.wavelength[self.goodrange])
        self.result = model.fit(self.flux[self.goodrange], params, x=self.wavelength[self.goodrange])
        print(self.result.fit_report())
        self.axes.plot(self.wavelength[self.goodrange], self.result.best_fit, 'r--x', label='Linear fit')
        self.plot_fit()

    def polynomial_fit(self, event):
        """Fits a polynomial in a specified region with a specified degree."""
        self.fitregion()
        model = mdl.PolynomialModel(degree=self.degree)
        params = model.guess(self.flux[self.goodrange],
                             x=self.wavelength[self.goodrange])
        self.result = model.fit(self.flux[self.goodrange], params, x=self.wavelength[self.goodrange])
        print(self.result.fit_report())
        self.axes.plot(self.wavelength[self.goodrange], self.result.best_fit, 'r--x', label='Polynomial fit')
        self.plot_fit()

    def exponential_fit(self, event):
        self.fitregion()
        model = mdl.ExponentialModel()
        params = model.guess(self.flux[self.goodrange], x=self.wavelength[self.goodrange])
        self.result = model.fit(self.flux[self.goodrange], params, x=self.wavelength[self.goodrange])
        print(self.result.fit_report())
        self.axes.plot(self.wavelength[self.goodrange], self.result.best_fit, 'r--x', label='Exponential fit')
        self.plot_fit()
    
    def powerlaw_fit(self, event):
        self.fitregion()
        model = mdl.PowerLawModel()
        params = model.guess(self.flux[self.goodrange], x=self.wavelength[self.goodrange])
        self.result = model.fit(self.flux[self.goodrange], params, x=self.wavelength[self.goodrange])
        print(self.result.fit_report())
        self.axes.plot(self.wavelength[self.goodrange], self.result.best_fit, 'r--x', label='Exponential fit')
        self.plot_fit()

    def sine_fit(self, event):
        self.fitregion()
        model = mdl.SineModel()
        params = model.guess(self.flux[self.goodrange], x=self.wavelength[self.goodrange])
        self.result = model.fit(self.flux[self.goodrange], params, x=self.wavelength[self.goodrange])
        print(self.result.fit_report())
        self.axes.plot(self.wavelength[self.goodrange], self.result.best_fit, 'r--x', label='Exponential fit')
        self.plot_fit()
    
    def gaussian_fit(self, event):
        """Fits a gaussian+linear fit to any given number of peaks chosen."""
        self.fitregion()
        if len(self.lines) > 3:
            self.gausscenter = np.array([self.lines[i+2][0] for i in range(len(self.lines[2:]))])
        else:
            self.gausscenter = self.lines[2][0]
        
        linear_model = mdl.LinearModel()
        if len(self.points) > 3:
            gauss_models = [mdl.GaussianModel(prefix=f'g{i}_') for i in range(len(self.gausscenter))]
            params = linear_model.guess(self.flux[self.goodrange], x=self.wavelength[self.goodrange])
            model = mdl.LinearModel()
            for i in range(len(gauss_models)):
                params.update(gauss_models[i].make_params())
                params[f'g{i}_center'].set(value=self.gausscenter[i],
                                           min=self.gausscenter[i]-0.005*self.gausscenter[i],
                                           max=self.gausscenter[i]+0.005*self.gausscenter[i])
                model += gauss_models[i]
        else:
            gauss_model = mdl.GaussianModel()
            params = gauss_model.make_params()
            params['center'].set(value=self.gausscenter)
            params.update(gauss_model.guess(self.flux[self.goodrange], x=self.wavelength[self.goodrange]))
            params.update(linear_model.guess(self.flux[self.goodrange], x=self.wavelength[self.goodrange]))
            model = gauss_model + linear_model
        self.result = model.fit(self.flux[self.goodrange], params, x=self.wavelength[self.goodrange])
        print(self.result.fit_report())
        self.axes.plot(self.wavelength[self.goodrange], self.result.best_fit, 'r--x', label='Gaussian fit')
        self.plot_fit()

    def exponentialgauss_fit(self, event):
        pass
    
    def lorentzian_fit(self, event):
        pass
    
    def harmonicoscillator_fit(self, event):
        pass
    
    def lognormal_fit(self, event):
        pass
    
    
    # FORMAT MENU
    def pop_up_window(self, title, windowsize, updatefunc, **labels):
        """Template function to create a pop up, two-fielded writable
        window to change dual values."""
        # Defining some initialization parameters
        row_amount = range(len(labels)+1)
        lbls, ents = {}, {}
        
        # Create the window itself.
        popupwindow = tk.Toplevel()
        popupwindow.title(title)
        popupwindow.geometry(windowsize)
        popupwindow.columnconfigure([0, 1], minsize=25, weight=1)
        popupwindow.rowconfigure(list(row_amount), minsize=25, weight=1)
        for i in range(len(row_amount)-1):
            lbls[f'lbl_{i}'] = tk.Label(master=popupwindow, text=labels[f'label{i+1}'])
            ents[f'ent_{i}'] = tk.Entry(master=popupwindow)
        
        # Creating the submit button.
        btn_submit = tk.Button(master=popupwindow, text='Submit',
                               command=updatefunc)

        # Setting up prompt and entry field positions.
        for i in range(len(row_amount)-1):
            lbls[f'lbl_{i}'].grid(row=row_amount[i], column=0)
            ents[f'ent_{i}'].grid(row=row_amount[i], column=1)
        btn_submit.grid(row=row_amount[-1], column=1)
        return popupwindow, ents

    def grid(self):
        """Draws a grid on the plot."""
        self.axes.grid()
        self.figure_canvas.draw()

    def tight_layout(self):
        """Tightens the layout of the plot, maxing out
        the function layout to the window."""
        self.figure.tight_layout()
        self.figure_canvas.draw()

    def x_lims(self):
        """Function that openes a window to input x limits
        to plot over."""
        def update_xlims():
            """Updates the x limits based on the given entry inputs."""
            xminimum = float(xlim_ents[f'ent_{0}'].get())
            xmaximum = float(xlim_ents[f'ent_{1}'].get())

            self.axes.set_xlim(xmin=xminimum, xmax=xmaximum)
            self.figure_canvas.draw()

            xlimswindow.destroy()
            xlimswindow.update()

        xlims_window = self.pop_up_window('X limits', '250x75', update_xlims,
                                    label1='X min:', label2='X max:')
        xlimswindow, xlim_ents = xlims_window

    def y_lims(self):
        """Function that openes a window to input y limits.
        """

        def update_ylims():
            """Updates the x limits based on the given input.
            """
            yminimum = float(ylim_ents[f'ent_{0}'].get())
            ymaximum = float(ylim_ents[f'ent_{1}'].get())
            
            self.axes.set_ylim(ymin=yminimum, ymax=ymaximum)
            self.figure_canvas.draw()

            ylimswindow.destroy()
            ylimswindow.update()

        ylims_window = self.pop_up_window('Y limits', '225x75', update_ylims,
                                    label1='Y min:', label2='Y max:')
        ylimswindow, ylim_ents = ylims_window

    def label_making(self):
        """Function that openes a window to input axis labels.
        """

        def update_label():
            """Updates the axis labels based on the given input.
            """
            x_label = self.axes.set_xlabel(label_ents[f'ent_{0}'].get(),
                                           size=float(label_ents[f'ent_{2}'].get()))
            y_label = self.axes.set_ylabel(label_ents[f'ent_{1}'].get(),
                                           size=float(label_ents[f'ent_{2}'].get()))
            self.figure_canvas.draw()

            labelwindow.destroy()
            labelwindow.update()

        labels_window = self.pop_up_window('Labels', '250x100', update_label,
                                    label1='X Label:', label2='Y Label:', label3='Font size:')
        labelwindow, label_ents = labels_window
    

    # HELP MENU
    def open_documentation(self):
        """Opens the documentation.
        """
        webbrowser.open('https://www.youtube.com/watch?v=dQw4w9WgXcQ')

    def contact_info(self):
        """Shows a window containing contact information.
        """
        contact_window = tk.Toplevel(self)
        contact_window.title('Contact Information')
        contact_window.geometry('250x75')
        contact_window.columnconfigure(0, minsize=25, weight=1)
        contact_window.rowconfigure([0, 1], minsize=25, weight=1)

        lbl_contact = tk.Label(text='Contact at: ', anchor=tk.CENTER,
                               font=('Helvetica', 14), master=contact_window)
        lbl_email = tk.Label(text='phdupont00@gmail.com ', anchor=tk.CENTER,
                               font=('Helvetica', 14), master=contact_window)
        lbl_contact.grid(row=0, column=0)
        lbl_email.grid(row=1, column=0)



if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('600x400')
    frame = App(root)
    frame.pack()
    frame.menu_bar()
    root.mainloop()