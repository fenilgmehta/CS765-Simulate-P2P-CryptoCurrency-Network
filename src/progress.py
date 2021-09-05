from gi.repository import Gtk, GLib

# REFER: https://www.geeksforgeeks.org/python-progressbar-in-gtk-3/
# REFER: https://zetcode.com/python/gtk/
class ProgressBarWindow(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title='Simulation Progress')
        self.set_border_width(10)

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)

        self.label = Gtk.Label()
        self.label.set_text('Initializing...')
        self.label.set_width_chars(50)
        vbox.pack_start(self.label, True, True, 0)

        # Create a ProgressBar
        self.progressbar = Gtk.ProgressBar()
        self.progressbar.set_size_request(width=150, height=-1)
        # self.progressbar.set_text('Initializing...')
        # self.progressbar.set_show_text(True)
        vbox.pack_start(self.progressbar, True, True, 0)

        # Create CheckButton with labels "Show text",
        # "Activity mode", "Right to Left" respectively
        button = Gtk.CheckButton(label="Activity mode")
        button.connect("toggled", self.on_activity_mode_toggled)
        vbox.pack_start(button, True, True, 0)

        # REFER: https://docs.gtk.org/glib/func.timeout_add.html
        self.timeout_id = GLib.timeout_add(500, self.on_timeout, None)
        self.activity_mode = False
        self.progress_percent: float = 0.0
        self.progress_label: str = 'Initializing...'

    def on_activity_mode_toggled(self, button):
        self.activity_mode = button.get_active()
        if self.activity_mode:
            self.progressbar.pulse()
        else:
            self.progressbar.set_fraction(0.0)

    def on_timeout(self, user_data):
        """
        Update value on the progress bar
        """
        if self.activity_mode:
            self.progressbar.pulse()
        else:
            # new_value = self.progressbar.get_fraction() + 0.01
            new_value = self.progress_percent
            if new_value > 1:
                new_value = 1
            self.progressbar.set_fraction(new_value)
            self.label.set_text(self.progress_label)
        return True

    @staticmethod
    def start_progressbar(win: 'ProgressBarWindow'):
        win.connect("destroy", Gtk.main_quit)
        win.show_all()
        Gtk.main()