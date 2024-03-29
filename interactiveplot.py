import os
import sys
import tkinter as tk
from tkinter import ttk
import webbrowser

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lmfit.models as mdl
from lmfit.model import save_modelresult, load_modelresult
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from sklearn.preprocessing import normalize
from scipy.optimize import curve_fit

matplotlib.use('TkAgg')

class App(tk.Frame):
    # Essentials
    def __init__(self, master):
        """Initializing the frame in which things are created onto in
        the window construct."""
        tk.Frame.__init__(self, master)
        operating_system = sys.platform
        style = ttk.Style()
        if operating_system == 'darwin':
            style.theme_use('aqua')
        self.master = master
        self.plottypevar = tk.StringVar(master=self.master)
        self.datacolorvar = tk.StringVar(master=self.master)
        self.yerrcolorvar = tk.StringVar(master=self.master)
        self.fitcolorvar = tk.StringVar(master=self.master)
        self.styles = [style for style in plt.style.available]
        self.stylevar = tk.StringVar(master=self.master)
        self.fitactivevar = tk.BooleanVar(master=self.master)
        self.fittypevar = tk.StringVar(master=self.master)
        self.master.title('Absorption/Emission line fitter')

    def dummy(self):
        """Dummy function to make non-implemented menu buttons
        do something instead of nothing."""

        window = tk.Toplevel(self)
        window.title('DUMMY')
        window.geometry('300x30')
        lbl_dummy = ttk.Label(master=window, text='BUTTON NOT IMPLEMENTED')
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
        fitmenu.add_command(label='Equivalent Width', command=self.choose_EW)
        fitmenu.add_separator()
        fitmenu.add_command(label='Save fit as...', command=lambda: self.save_fit(self.result))
        fitmenu.add_command(label='Load fit...', command=self.load_fit)
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
        formatmenu.add_separator()
        formatmenu.add_command(label='Color', command=self.color_change)
        formatmenu.add_command(label='Style', command=self.style_change)


        # Creating the options for the "help" drop down menu.
        helpmenu.add_command(label='Controls', command=self.dummy)
        helpmenu.add_command(label='Documentation', command=self.dummy)#open_documentation)
        helpmenu.add_command(label='Contact', command=self.dummy)#contact_info)

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
        plt.style.use(self.stylevar.get())  
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.figure_canvas = FigureCanvasTkAgg(self.figure, master=master)
        self.figure_canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.figure_canvas, master, pack_toolbar=False)
        self.toolbar.update()

        self.axes = self.figure.add_subplot()
        data_color = self.datacolorvar.get()
        yerr_color = self.yerrcolorvar.get()
        self.axes.step(self.x, self.y, label='Data', color=data_color)    
        self.axes.plot(self.x, self.yerr, '--', label='y-error', color=yerr_color)
        self.axes.legend()    

        self.figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH,
                                                expand=True)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.BOTH)

    def reset(self, event):
        """Function to reset all the parameters and changes made to the plot.
        """
        print('Resetting...')
        self.gridvar.set(False)
        self.figure_canvas.get_tk_widget().destroy()
        self.toolbar.destroy()
        self.fitwindow.destroy()
        self.vlinelist = []
        self.points = []
        self.fitactivevar.set(False)
        self.x = np.array(self.df[self.xcol]).copy()*self.scalingfactor_x
        self.y = np.array(self.df[self.ycol]).copy()*self.scalingfactor_y
        self.yerr = np.array(self.df[self.yerrcol]).copy()*self.scalingfactor_y
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
                                                      title = 'select a file')
        self.df = pd.read_table(self.filename, delim_whitespace=True, header=None)
        self.NAME = os.path.basename(os.path.normpath(os.path.splitext(self.filename)[0]))
        self.df.name = self.NAME
        self.column_amt = len(self.df.columns)
        print(self.df.head())
        def create_data():
            if options_ents['ent_0'].get():
                self.scalingfactor_x = float(options_ents['ent_0'].get())
            else: 
                self.scalingfactor_x = 1
            if options_ents['ent_1'].get():
                self.scalingfactor_y = float(options_ents['ent_1'].get())
            else:
                self.scalingfactor_y = 1
                
            self.xcol, self.ycol = int(options_ents['ent_2'].get()), int(options_ents['ent_3'].get())
            self.yerrcol = int(options_ents['ent_4'].get())
            options_window.destroy()
            options_window.update()
            self.x = np.array(self.df[self.xcol]).copy()*self.scalingfactor_x
            self.y = np.array(self.df[self.ycol]).copy()*self.scalingfactor_y
            self.yerr = np.array(self.df[self.yerrcol].copy())*self.scalingfactor_y
            self.datacolorvar.set('blue')
            self.yerrcolorvar.set('green')
            self.fitcolorvar.set('red')
            self.stylevar.set('default')
            self.fitactivevar.set(False)
            self.draw_figure(master=self.master)
            
    
        options_window = self.pop_up_window('File options', '375x200', create_data,
                                    label1='X-axis scaling factor (leave blank if none):',
                                    label2='Y-axis scaling factor (leave blank if none):',
                                    label3=f'Choose x-axis data column [0-{self.column_amt-1}]:',
                                    label4=f'Choose y-axis data column [0-{self.column_amt-1}]:',
                                    label5=f'Choose y-error data column [0-{self.column_amt-1}]:')
        options_window, options_ents = options_window
        
    def open_files(self):
        pass

    def save_plot(self):
        """Saves figure in the active script folder with filename as name
        """
        self.figure.savefig(f'{self.NAME}.pdf')

    def save_plot_as(self):
        """Saves the current plot as user selected filename in user selected folder.
        """
        save_filename = tk.filedialog.asksaveasfilename(initialfile='Untitled.pdf',
                                                        defaultextension='.pdf',
                                                        filetypes=[('All Files', '*.*'),
                                                                   ('test','*.png')])
        self.figure.savefig(save_filename, dpi=300)

    def load_fit(self):
        """Loads saved fit result JSON from file.
        """
        saved_fit = tk.filedialog.askopenfilename(initialdir=os.listdir(),
                                                      title = 'select a file',
                                                      filetypes = (('All Files', '*.*'),))
        
        self.result = load_modelresult(saved_fit)
        self.rangemin, self.rangemax = self.result.userkws['x'][-1], self.result.userkws['x'][0]
        self.goodregion = np.array([self.rangemin, self.rangemax])
        self.goodrange = np.where((self.x >= self.goodregion[0]) &
                                  (self.x <= self.goodregion[1]))
        self.plot_fit()
        
    def save_fit(self, current_fit_result):
        """Saves the current fit as a JSON computer-readable file.

        Args:
            current_fit_result (ModelResult): result from currently fitted model.
        """
        fit_report_filename = tk.filedialog.asksaveasfilename(initialfile='fit.sav',
                                                              defaultextension='.sav',
                                                              filetypes=[('All Files', '*.*')])
        
        save_modelresult(current_fit_result, fit_report_filename)
        
    def save_fit_report(self, current_fit_result):
        """Saves the fit report to a user readable .txt file.

        Args:
            current_fit_result (ModelResult): current model result to create a fit report from.
        """
        fit_report_filename = tk.filedialog.asksaveasfilename(initialfile='fit_report.txt',
                                                              defaultextension='.txt',
                                                              filetypes=[('All Files', '*.*')])
        with open(fit_report_filename, 'w') as fr:
            fr.write(current_fit_result.fit_report())

    # EDIT MENU
        
    def zoom(self):
        """Implements the toolbar zoom to a custom button.
        """
        self.figure_canvas.mpl_connect('button_release_event', self.toolbar.zoom)

    def smooth_data(self):
        """Smooths the data using box smoothing.
        """
        self.initialize_fit_frame()
        def smooth():
            box_pts = int(smooth_ent[f'ent_{0}'].get())
            box = np.ones(box_pts)/box_pts
            self.y = np.convolve(self.y, box, mode='same')
            self.draw_figure(master=self.fitwindow)
            self.fitwindow.focus_set()
            self.y = np.array(self.df[self.ycol])

            smoothwindow.destroy()
            smoothwindow.update()
    
        smooth_window = self.pop_up_window('Smoothing level', '250x50',
                                           smooth, label1='Enter box smoothing')
        smoothwindow, smooth_ent = smooth_window
        
    def cut_data(self):
        """Cuts a user-selected set of data from the plot.
        """
        self.initialize_fit_frame()
        def cut(event):
            self.fitregion()
            self.y = np.delete(self.y, self.goodrange)
            self.x = np.delete(self.x, self.goodrange)
            self.line_removal()
            self.draw_figure(master=self.fitwindow)
            self.fitwindow.focus_set()
        self.fitwindow.bind('<Return>', cut)

    def color_change(self):
        def choose_color():
            dat_col = colorchange_ent['ent_0'].get()
            yerr_col = colorchange_ent['ent_1'].get() 
            if self.fitcolorvar.get():
                fit_col = colorchange_ent['ent_2'].get()
                self.fitcolorvar.set(fit_col)
                
            if dat_col:
                self.datacolorvar.set(dat_col)
                
            if yerr_col:
                self.yerrcolorvar.set(yerr_col)
            
            if self.fitactivevar.get():
                self.temp_save_fit(self.result)
                self.initialize_fit_frame()
                self.temp_load_fit(self.tempfitfile)
            else:
                self.draw_figure(master=self.master)
                
            colorchangewindow.destroy()
            colorchangewindow.update()
            
        if self.fitcolorvar.get():
            colorchangewindow, colorchange_ent = self.pop_up_window('Choose plot color',
                                                                '300x100',
                                                                choose_color,
                                                                label1='Data color [color name/hex code]',
                                                                label2='Y-error color [color name/hex code]',
                                                                label3='Fit color [color name/hex code]')
        else:
            colorchangewindow, colorchange_ent = self.pop_up_window('Choose plot color',
                                                                    '300x75',
                                                                    choose_color,
                                                                    label1='Data color [color name/hex code]',
                                                                    label2='Y-error color [color name/hex code]')

      
    def style_change(self):
        print(self.styles)
        def choose_style():
            style = stylechange_ent['ent_0'].get()
            if style:
                self.stylevar.set(style)
                
            if self.fitactivevar.get():
                self.temp_save_fit(self.result)
                self.initialize_fit_frame()
                self.temp_load_fit(self.tempfitfile)
            else:
                self.draw_figure(master=self.master)
                
            stylechangewindow.destroy()
            stylechangewindow.update()
        stylechangewindow, stylechange_ent = self.pop_up_window('Choose matplotlib style',
                                                                '500x50', choose_style,
                                                                label1='Write matplotlib style [see console for styles]')
      
    def normalize_data(self):
        """Normalizes the full data_set then resets the y_values after drawing it.
        """
        self.initialize_fit_frame()
        self.y = normalize(self.y.reshape(1, -1))[0]
        self.draw_figure(master=self.fitwindow)
        self.y = np.array(self.df[self.ycol].copy())*self.scalingfactor_y
        self.fitwindow.focus_set()
    
    def normalize_section(self):
        """Normalizes a section of the data using sklearn preprocessing. The area is chosen by
        marking vertical lines at the x-axis limits.
        """
        self.initialize_fit_frame()
        def norm(event):
            self.fitregion()
            self.y[self.goodrange] = normalize(self.y[self.goodrange].reshape(1, -1))[0]
            self.line_removal()
            self.draw_figure(master=self.fitwindow)
            self.y = np.array(self.df[self.ycol]).copy()*self.scalingfactor_y
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
        """Removes all vertical lines on the plot and resets their respective stored values.
        """
        for i in range(len(self.vlinelist)): 
            self.vlinelist.pop(0).remove()
            self.lines = np.delete(self.lines, 0)
            self.points.pop(0)

    def set_xlims(self, event):
        """Sets the x-axis limits for plotting purposes from the two fit function limits.
        """
        if len(self.points) > 1 and len(self.points) <= 2:
            xlimmin, xlimmax = self.points[0][0], self.points[1][0]
            self.axes.set_xlim(xlimmin-0.01*xlimmin,
                               xlimmax+0.01*xlimmax)
            print(xlimmin, xlimmax)
        self.figure_canvas.draw()

    def fitregion(self):
        """Determines the limits on the x-axis of where to fit, based on user chosen lines.
        """
        self.lines = np.array(self.points)
        if self.lines[0][0] < self.lines[1][0]:
            self.rangemin = self.lines[0][0]
            self.rangemax = self.lines[1][0]
        elif self.lines[0][0] > self.lines[1][0]:
            self.rangemin = self.lines[1][0]
            self.rangemax = self.lines[0][0]
        self.goodregion = np.array([self.rangemin, self.rangemax])
        self.goodrange = np.where((self.x >= self.goodregion[0]) &
                                  (self.x <= self.goodregion[1]))

    def initialize_fit_frame(self):
        """Initializes the frame that is used for fitting, drawing the vertical lines for choosing the fit
        and for visualising the fit onto the data.
        """
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
        """Initializes the linear fit.
        """
        self.fitactivevar.set(True)
        self.initialize_fit_frame()
        print(self.fitactivevar.get())
        self.fitwindow.bind('<Return>', self.linear_fit)

    def choose_polynomial(self):
        """Initializes the polynomial fit.
        """
        self.fitactivevar.set(True)
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
        """Initializes the exponential fit.
        """
        self.fitactivevar.set(True)
        self.initialize_fit_frame()
        self.fitwindow.bind('<Return>', self.exponential_fit)
    
    def choose_powerlaw(self):
        """Initializes the powerlaw fit.
        """
        self.fitactivevar.set(True)
        self.initialize_fit_frame()
        self.fitwindow.bind('<Return>', self.powerlaw_fit)
    
    def choose_sine(self):
        """Initializes the sinusodial fit.
        """
        self.initialize_fit_frame()
        self.fitwindow.bind('<Return>', self.sine_fit)

    def choose_gauss(self):
        """Initializes the gaussian fit.
        """
        self.fitactivevar.set(True)
        self.initialize_fit_frame()
        self.fitwindow.bind('<Return>', self.gaussian_fit)
    
    def choose_exponentialgauss(self):
        """Initializes the exponential gaussian fit"""
        self.fitactivevar.set(True)
        self.initialize_fit_frame()
        self.fitwindow.bind('<Return>', self.exponentialgauss_fit)
        
    def choose_lorentzian(self):
        """Initializes the exponential gaussian fit"""
        self.fitactivevar.set(True)
        self.initialize_fit_frame()
        self.fitwindow.bind('<Return>', self.lorentzian_fit)
        
    def choose_harmonicoscillator(self):
        """Initializes the exponential gaussian fit"""
        self.fitactivevar.set(True)
        self.initialize_fit_frame()
        self.fitwindow.bind('<Return>', self.harmonicoscillator_fit)
        
    def choose_lognormal(self):
        """Initializes the exponential gaussian fit"""
        self.fitactivevar.set(True)
        self.initialize_fit_frame()
        self.fitwindow.bind('<Return>', self.lognormal_fit)
    
    def choose_EW(self):
        """Initializes the equivalent width calculation/fitting
        """
        self.fitactivevar.set(True)
        self.initialize_fit_frame()
        self.equivalent_width()
    
    def plot_fit(self):
        fit_color = self.fitcolorvar.get()
        fit_type = self.fittypevar.get()
        self.axes.plot(self.x[self.goodrange], self.result.best_fit, '--x',
                       label=f'{fit_type} fit', color=fit_color)
        self.line_removal()
        self.axes.legend()
        self.figure_canvas.draw()
        self.vlinelist = []
        self.points = []
        
    def temp_save_fit(self, current_fit_result):
        self.tempfitfile = os.getcwd()+r'\\fit.sav'
        save_modelresult(current_fit_result, self.tempfitfile)

    def temp_load_fit(self, saved_fit):
        self.result = load_modelresult(saved_fit)
        os.remove(self.tempfitfile)
        self.rangemin, self.rangemax = self.result.userkws['x'][-1], self.result.userkws['x'][0]
        self.goodregion = np.array([self.rangemin, self.rangemax])
        self.goodrange = np.where((self.x >= self.goodregion[0]) &
                                  (self.x <= self.goodregion[1]))
        self.plot_fit()
        
        # Functions that creates and fits the different models
    def linear_fit(self, event):
        """Fits a simple linear line to a chosen region."""
        self.fitregion()
        model = mdl.LinearModel()
        params = model.guess(self.y[self.goodrange],
                             x=self.x[self.goodrange])
        self.result = model.fit(self.y[self.goodrange], params, x=self.x[self.goodrange])
        print(self.result.fit_report())
        self.fittypevar.set('Linear')
        self.plot_fit()

    def polynomial_fit(self, event):
        """Fits a polynomial in a specified region with a specified degree."""
        self.fitregion()
        model = mdl.PolynomialModel(degree=self.degree)
        params = model.guess(self.y[self.goodrange],
                             x=self.x[self.goodrange])
        self.result = model.fit(self.y[self.goodrange], params, x=self.x[self.goodrange])
        print(self.result.fit_report())
        self.fittypevar.set('Polynomial')
        self.plot_fit()

    def exponential_fit(self, event):
        """Fits an exponential function to the data given user selected x-axis limits.
        """
        self.fitregion()
        model = mdl.ExponentialModel()
        params = model.guess(self.y[self.goodrange], x=self.x[self.goodrange])
        self.result = model.fit(self.y[self.goodrange], params, x=self.x[self.goodrange])
        print(self.result.fit_report())
        self.fittypevar.set('Exponential')
        self.plot_fit()
    
    def powerlaw_fit(self, event):
        """Fits a powerlaw to the data given user selected x-axis limits.
        """
        self.fitregion()
        model = mdl.PowerLawModel()
        params = model.guess(self.y[self.goodrange], x=self.x[self.goodrange])
        self.result = model.fit(self.y[self.goodrange], params, x=self.x[self.goodrange])
        print(self.result.fit_report())
        self.fittypevar.set('Powerlaw')
        self.plot_fit()

    def sine_fit(self, event):
        """Fits a sine function to the data given user selected x-axis limits.
        """
        self.fitregion()
        model = mdl.SineModel()
        params = model.guess(self.y[self.goodrange], x=self.x[self.goodrange])
        self.result = model.fit(self.y[self.goodrange], params, x=self.x[self.goodrange])
        print(self.result.fit_report())
        self.fittypevar.set('Sine')
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
            params = linear_model.guess(self.y[self.goodrange], x=self.x[self.goodrange])
            model = linear_model
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
            params.update(gauss_model.guess(self.y[self.goodrange], x=self.x[self.goodrange]))
            params.update(linear_model.guess(self.y[self.goodrange], x=self.x[self.goodrange]))
            model = gauss_model + linear_model
        self.result = model.fit(self.y[self.goodrange], params, x=self.x[self.goodrange])
        print(self.result.fit_report())
        self.fittypevar.set('Gaussian')
        self.plot_fit()

    def exponentialgauss_fit(self, event):
        """Fits an exponential gaussian fit+exponential normal fit to any given number of peaks chosen."""
        self.fitregion()
        if len(self.lines) > 3:
            expgausscenter = np.array([self.lines[i+2][0] for i in range(len(self.lines[2:]))])
        else:
            expgausscenter = self.lines[2][0]
        
        exp_model = mdl.ExponentialModel()
        if len(self.points) > 3:
            expgauss_models = [mdl.ExponentialGaussianModel(prefix=f'g{i}_') for i in range(len(expgausscenter))]
            params = exp_model.guess(self.y[self.goodrange], x=self.x[self.goodrange])
            model = exp_model
            for i in range(len(expgauss_models)):
                params.update(expgauss_models[i].make_params())
                params[f'g{i}_center'].set(value=expgausscenter[i],
                                           min=expgausscenter[i]-0.005*expgausscenter[i],
                                           max=expgausscenter[i]+0.005*expgausscenter[i])
                model += expgauss_models[i]
        else:
            gauss_model = mdl.ExponentialGaussianModel()
            params = gauss_model.make_params()
            params['center'].set(value=expgausscenter)
            params.update(gauss_model.guess(self.y[self.goodrange], x=self.x[self.goodrange]))
            params.update(exp_model.guess(self.y[self.goodrange], x=self.x[self.goodrange]))
            model = gauss_model + exp_model
        self.result = model.fit(self.y[self.goodrange], params, x=self.x[self.goodrange])
        print(self.result.fit_report())
        self.fittypevar.set('Exponential Gaussian')
        self.plot_fit()
    
    def lorentzian_fit(self, event):
        pass
    
    def harmonicoscillator_fit(self, event):
        pass
    
    def lognormal_fit(self, event):
        pass
    
    
    # The following functions are all used to calculate the equivalent width
    def linear(self, x, a, b):
        """Creates a linear line.

        Args:
            x (int, float, array): input value or array
            a (int, float): slope coefficient
            b (int, float): y intersect coefficient

        Returns:
            int, float, array: y value(s) for the linear line
        """
        return a*x + b
            
    def choose_line(self, event):
        """Function to select a line to calculate the equivalent width on.
        Zooms into the area around the line too.
        """
        self.lines = np.array(self.points)
        self.selected_line = self.lines[0][0]
        self.axes.set_xlim(self.selected_line-self.normrange, self.selected_line+self.normrange)
        self.plot_fit()
        self.fitwindow.bind('<Return>', self.choose_normalization)
    
    def choose_normalization(self, event):
        """Function to select areas around the line to normalize to, then normalizes.
        Zooms into the newly normalized area.
        """
        self.lines = np.array(self.points)
        norm_left1, norm_left2 = self.lines[0][0], self.lines[1][0]
        norm_right1, norm_right2 = self.lines[2][0], self.lines[3][0]
        df_x = self.df[self.xcol]
        df_norm_fit = self.df[(df_x > norm_left1) & (df_x < norm_left2) | (df_x > norm_right1) & (df_x < norm_right2)]
        param_guess = [0., 1.]
        print(df_norm_fit)
        val, cov = curve_fit(self.linear, df_norm_fit[self.xcol]*self.scalingfactor_x, 
                             df_norm_fit[self.ycol]*self.scalingfactor_y,
                            p0=param_guess)
        self.a, self.b = val[0], val[1]
        self.norm = self.y/(self.linear(df_x, self.a, self.b))
        self.normerr = self.yerr/(self.linear(df_x, self.a, self.b))
        self.y = np.array(self.norm)
        self.yerr = np.array(self.normerr)
        self.plot_fit()
        self.initialize_fit_frame()
        minx, maxx = self.selected_line - self.normrange, self.selected_line + self.normrange
        self.axes.set_xlim(minx, maxx)
        minyidx = int(np.where(np.abs(self.x-(minx))==np.abs(self.x-(minx)).min())[0])
        maxyidx = int(np.where(np.abs(self.x-(maxx))==np.abs(self.x-(maxx)).min())[0])
        miny, maxy = 0.5*self.y[minyidx], 1.5*self.y[maxyidx]
        self.axes.set_ylim(miny, maxy)
        self.fitwindow.bind('<Return>', self.choose_line_area)
            
    def choose_line_area(self, event):
        """Function to choose the edges of the selected line, then calculates the equivalent width.
        """
        self.lines = np.array(self.points)
        line_left, line_right = self.lines[0][0], self.lines[1][0]
        
        df_x = self.df[self.xcol]
        self.df_moment = self.df[(df_x > line_left) & (df_x < line_right)]
        df_moment_x = self.df_moment[self.xcol]
        df_moment_y = self.df_moment[self.ycol]
        df_moment_yerr = self.df_moment[self.yerrcol]
        self.profile = 1.-((df_moment_y*self.scalingfactor_y)/(self.linear((df_moment_x*self.scalingfactor_x), self.a, self.b)))
        M1 = np.sum((df_moment_x*self.scalingfactor_x)*self.profile/(df_moment_yerr*self.scalingfactor_y)**2)/np.sum(self.profile/(df_moment_yerr*self.scalingfactor_y)**2)
        EW = self.lstep*np.sum(self.profile)
        if self.scalingfactor_y == 1:
            self.scalingfactor_yerr = EW/np.mean(df_moment_yerr) #rough automatic scaling for yerr
        else:
            self.scalingfactor_yerr = self.scalingfactor_y
        EWerr = self.lstep*np.sqrt(np.sum((df_moment_yerr*self.scalingfactor_yerr)**2))
        print('Wavelength of the line:',f'{M1:.2f}','AA')
        print('EW: ',f'{EW:.2f}','AA +/- ',f'{EWerr:.2f}','AA')
        self.plot_fit()
        
    def equivalent_width(self):
        """Calculates the equivalent width using the functions choose_line,
        choose_normalization and choose_line_area
        """
        self.normrange = 100.
        self.lstep = self.x[1]-self.x[0]
        self.fitwindow.bind('<Return>', self.choose_line)

        
        
    # FORMAT MENU
    def pop_up_window(self, title, windowsize, updatefunc, **labels):
        """Template function to create a pop up, two-fielded writable
        window to change dual values."""
        # Defining some initialization parameters
        row_amount = range(len(labels)+1)
        lbls, ents = {}, {}
        
        # Create the window itself.
        popupwindow = tk.Toplevel(takefocus=True, background='#ececec')
        popupwindow.title(title)
        popupwindow.geometry(windowsize)
        popupwindow.columnconfigure([0, 1], minsize=25, weight=1)
        popupwindow.rowconfigure(list(row_amount), minsize=25, weight=1)
        for i in range(len(row_amount)-1):
            lbls[f'lbl_{i}'] = ttk.Label(master=popupwindow, text=labels[f'label{i+1}'])#, background='#ececec')
            ents[f'ent_{i}'] = ttk.Entry(master=popupwindow, background='#ececec')
        
        # Creating the submit button.
        btn_submit = ttk.Button(master=popupwindow, text='Submit',
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
            self.axes.legend(fontsize=float(label_ents[f'ent_{2}'].get())*0.9)
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

        lbl_contact = ttk.Label(text='Contact at: ', anchor=tk.CENTER,
                               font=('Helvetica', 14), master=contact_window)
        lbl_email = ttk.Label(text='phdupont00@gmail.com ', anchor=tk.CENTER,
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