package sfbPlot;

/**
 * Title:        SuperFatboyPlot.java
 * Version:      (see rcsID)
 * Copyright:    Copyright (c) 2006
 * Author:       Craig Warner
 * Company:      University of Florida
 * Description:  flam2helper main class
 */

import javaUFLib.*;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import java.util.*;
import java.io.*;
import java.net.URL;
import javax.swing.border.*;

public class SuperFatboyPlot extends JFrame {
    public static final
        String rcsID = "$Name:  $ $Id: SuperFatboyPlot.java,v 1.2 2016/07/14 18:17:46 warner Exp $";

    protected ArrayList<UFRSSPlot> graphPanels;
    protected ArrayList<UFRSSPlotPanel> thePanels;

    protected String logName = "current", prefsName = ".sfbplot";
    protected String defPath = ".";
    protected int npanels;
    protected JTabbedPane graphTabs; 
    protected JPanel overallGraphPanel;
    protected LinkedHashMap colors;

    public SuperFatboyPlot() { 
      super("SuperFATBOY Plot");
      setupGUI(null);
      setPlots();
      startPlots();
    }

    public SuperFatboyPlot(String[] args) {
      super("SuperFATBOY Plot");
      setupGUI(args);
      setPlots();
      startPlots();
    }

    public void setupGUI(String[] args) {
      graphTabs = new JTabbedPane();
      overallGraphPanel = new JPanel();
      overallGraphPanel.setLayout(new BorderLayout());
      overallGraphPanel.add(graphTabs, BorderLayout.CENTER);

      graphPanels = new ArrayList();
      thePanels = new ArrayList();

      if (args.length == 0) {
        UFRSSPlot plot = new UFRSSPlot();
        graphPanels.add(plot);
        thePanels.add(new UFRSSPlotPanel(plot));
        npanels = 1;
      } else {
	for (int j = 0; j < args.length; j++) {
	  UFRSSPlot plot = new UFRSSPlot(args[j]);
	  graphPanels.add(plot);
          thePanels.add(new UFRSSPlotPanel(plot));
	}
	npanels = args.length;
      }
    }

    public void setPlots() {
        graphTabs.removeAll();
        for (int j = 0; j < npanels; j++) {
          graphTabs.add(thePanels.get(j), thePanels.get(j).panelTitle);
        }

        setSize(942,726);
        Container content = getContentPane();
        content.setBackground(Color.black);
        content.setLayout(new BorderLayout());

        addWindowListener(new WindowAdapter() {
          public void windowClosing(WindowEvent e) {
            SuperFatboyPlot.this.dispose();
          }
        });

        addComponentListener(new ComponentAdapter() {
          public void componentResized(ComponentEvent ev) {
            SuperFatboyPlot sfp = (SuperFatboyPlot)ev.getSource();
	    for (int j = 0; j < npanels; j++) {
	      sfp.thePanels.get(j).thePlot.resizePlot(sfp.getWidth()-337, sfp.getHeight()-183);
	      sfp.thePanels.get(j).plot();
	    }
          }
        });

	content.add(overallGraphPanel, BorderLayout.CENTER);
	pack();
        setVisible(true);
        setDefaultCloseOperation(3);
    }

    public void startPlots() {
        for (int j = 0; j < npanels; j++) {
	  thePanels.get(j).plot();
        }
    }

/*
    protected void processWindowEvent(WindowEvent wev) {
        System.exit(0);
    }
*/


    public class UFRSSPlotPanel extends JPanel {
      protected JPanel leftPanel;
      protected SpringLayout leftLayout;

      protected JTextField titleField, xtitleField, ytitleField, charSizeField;
      protected JTextField xMinField, xMaxField, yMinField, yMaxField;
      protected JTextField bgColorField, axesColorField;
      protected JButton plotButton, colorChooser, saveColorButton;
      protected JButton addFileButton, renameTabButton, newPanelButton;
      protected UFRSSPlot thePlot;
      protected JLabel x1l,x2l,y1l,y2l,dxl,dyl;
      protected UFColorCombo[] dragObjectBox;
      protected int numSpectra;
      public String panelTitle = "Plot";

      protected JRadioButton xLin, xLog, yLin, yLog;
      protected ButtonGroup plotType, xLinLog, yLinLog;
      protected JButton multiButton, optsButton, linFitButton;
      protected JButton quitButton, addTextButton, UFButton;
      protected int xticks=0, yticks=0, xminor=0, yminor=0;
      protected float xtickInt=0, ytickInt=0, xtickLen=0, ytickLen=0, symsize=0;
      protected float[] xtickVals, ytickVals, xmargin, ymargin, position;
      protected String[] xtickNames, ytickNames;
      protected String fontName="";
      protected JLabel coordLabel;
      public boolean multiMode = false;

      public UFRSSPlotPanel() {
	super();
      }

      public UFRSSPlotPanel(final UFRSSPlot plot) {
        super();
        thePlot = plot;
        thePlot.setPlotPanel(this);
        numSpectra = thePlot.numSpectra;
        setupComponents();
	setupLeftPanel();
        drawComponents();
        startPlot();
      }

      public void setupComponents() {
        numSpectra = thePlot.numSpectra;
        leftPanel = new JPanel();

	colors = new LinkedHashMap();
        // see if colors are stored in config file
        try {
            String home = UFExecCommand.getEnvVar("HOME");
            File f = new File(home+"/"+prefsName);
            if (f.exists()) {
                BufferedReader br = new BufferedReader(new FileReader(f));
                while (br.ready()) {
                    String s = br.readLine();
                    if (s != null && !s.trim().equals("")) {
                        StringTokenizer st = new StringTokenizer(s);
                        if (st.countTokens() == 2) {
                            String sname = st.nextToken().trim();
			    String rgb = st.nextToken().trim();
			    colors.put(sname.replaceAll("_"," "), rgb);
                        }
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("SuperFatboyPlot> "+e.toString());
        }

	addFileButton = new JButton("Add File");
        addFileButton.addActionListener(new ActionListener() {
           public void actionPerformed(ActionEvent ev) {
              JFileChooser jfc = new JFileChooser(defPath);
              int returnVal = jfc.showOpenDialog((Component)ev.getSource());
              if (returnVal == JFileChooser.APPROVE_OPTION) {
                defPath = jfc.getCurrentDirectory().getAbsolutePath();
		String fitsFile = jfc.getSelectedFile().getAbsolutePath();
		thePlot.addFitsFile(fitsFile);
		setupLeftPanel();
		revalidate();
		repaint();
	        startPlot();
		thePlot.updatePlot();
              }
           }
        });

	renameTabButton = new JButton("Rename");
	renameTabButton.addActionListener(new ActionListener() {
           public void actionPerformed(ActionEvent ev) {
    	      String title = JOptionPane.showInputDialog("Enter new tab title...", null);
	      panelTitle = title;
	      graphTabs.setTitleAt(graphTabs.getSelectedIndex(), title);
           }
        });

	newPanelButton = new JButton("New Tab");
	newPanelButton.addActionListener(new ActionListener() {
           public void actionPerformed(ActionEvent ev) {
              UFRSSPlot plot = new UFRSSPlot();
              graphPanels.add(plot);
              thePanels.add(new UFRSSPlotPanel(plot));
	      graphTabs.add(thePanels.get(npanels), thePanels.get(npanels).panelTitle);
              npanels++;
	      startPlots();
           }
        });

	plotButton = new JButton("Plot");
	plotButton.addActionListener(new ActionListener() {
	   public void actionPerformed(ActionEvent ev) {
	      startPlot();
	      thePlot.updatePlot();
	   }
	});

	/* Color Chooser */
	colorChooser = new JButton("Color Chooser");
	colorChooser.addActionListener(new ActionListener() {
	   public void actionPerformed(ActionEvent ev) {
	      final JDialog retVal = new JDialog();
              retVal.setModal(false);
              retVal.setAlwaysOnTop(true);
              retVal.setSize(200,40*(numSpectra+2)+10);
	      if (numSpectra > 16) retVal.setSize(200, 770);
              retVal.setLayout(new GridLayout(0,1));
	      JPanel thePanel = new JPanel();
	      thePanel.setLayout(new GridLayout(0,1));
              for (int i=0; i<numSpectra; i++) {
		final int myI = i;
		final JLabel showLabel = new JLabel(thePlot.spectra.get(i).spectrumName);
		final JButton colorButton = new JButton();
		final Color tempColor = getColor(thePlot.spectra.get(i).colorField.getText());
		colorButton.setBackground(tempColor);
		colorButton.addActionListener(new ActionListener(){
                    public void actionPerformed(ActionEvent ae) {
                        Color c = JColorChooser.showDialog(retVal,"Choose Color",tempColor);
                        if (c != null) {
			   thePlot.spectra.get(myI).colorField.setForeground(c);
			   thePlot.spectra.get(myI).colorField.setText(""+c.getRed()+","+c.getGreen()+","+c.getBlue());
			   colorButton.setBackground(c);
                           getParent().repaint();
                           plotButton.doClick();
			}
                    }
                });
		
		JPanel pan = new JPanel();
		pan.setLayout(new RatioLayout());
		pan.add("0.01,0.01;0.80,0.99",showLabel);
		pan.add("0.81,0.01;0.19,0.99",colorButton);
		thePanel.add(pan);
		//retVal.add(pan);
		//retVal.setVisible(true);
	      }
	      final JLabel bgshowLabel = new JLabel("BG Color");
	      final JButton bgcolorButton = new JButton();
	      final Color bgtempColor = getColor(bgColorField.getText());
	      bgcolorButton.setBackground(bgtempColor);
	      bgcolorButton.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent ae) {
		   Color c = JColorChooser.showDialog(retVal,"Choose Color",bgtempColor);
		   if (c != null) {
		      bgColorField.setText(""+c.getRed()+","+c.getGreen()+","+c.getBlue());
		      bgcolorButton.setBackground(c);
		      getParent().repaint();
                      plotButton.doClick();
		   }
		}
	      });
              JPanel bgpan = new JPanel();
              bgpan.setLayout(new RatioLayout());
              bgpan.add("0.01,0.01;0.80,0.99",bgshowLabel);
              bgpan.add("0.81,0.01;0.19,0.99",bgcolorButton);
              //retVal.add(bgpan);
              thePanel.add(bgpan);
              final JLabel axshowLabel = new JLabel("Axes Color");
              final JButton axcolorButton = new JButton();
              final Color axtempColor = getColor(axesColorField.getText());
              axcolorButton.setBackground(axtempColor);
              axcolorButton.addActionListener(new ActionListener(){
                public void actionPerformed(ActionEvent ae) {
                   Color c = JColorChooser.showDialog(retVal,"Choose Color",axtempColor);
                   if (c != null) {
                      axesColorField.setText(""+c.getRed()+","+c.getGreen()+","+c.getBlue());
                      axcolorButton.setBackground(c);
                      getParent().repaint();
		      plotButton.doClick();
		   }
                }
              });
              JPanel axpan = new JPanel();
              axpan.setLayout(new RatioLayout());
              axpan.add("0.01,0.01;0.80,0.99",axshowLabel);
              axpan.add("0.81,0.01;0.19,0.99",axcolorButton);
              //retVal.add(axpan);
              thePanel.add(axpan);
	      retVal.add(new JScrollPane(thePanel));
              retVal.setVisible(true);
	   }
	});

	bgColorField = new JTextField(8);
	if (colors.containsKey("BG_Color")) {
	   String temp = (String)colors.get("BG_Color");
	   bgColorField.setText(temp);
	} else bgColorField.setText("255,255,255");
	bgColorField.addFocusListener(new FocusListener() {
	   public void focusGained(FocusEvent fe) {
	   }

	   public void focusLost(FocusEvent fe) {
	      Color tempColor = getColor(bgColorField.getText());
	      if (tempColor != null) {
		plotButton.doClick();
	      }
	   }
	});

	axesColorField = new JTextField(8);
        if (colors.containsKey("Axes_Color")) {
           String temp = (String)colors.get("Axes_Color");
           axesColorField.setText(temp);
        } else axesColorField.setText("0,0,0");
        axesColorField.addFocusListener(new FocusListener() {
           public void focusGained(FocusEvent fe) {
           }

           public void focusLost(FocusEvent fe) {
              Color tempColor = getColor(axesColorField.getText());
              if (tempColor != null) {
                plotButton.doClick();
              }
           }
        });

	/* Save Color Button */
	saveColorButton = new JButton("Save Colors");
	saveColorButton.addActionListener(new ActionListener() {
	   public void actionPerformed(ActionEvent ev) {
	      try {
		String home = UFExecCommand.getEnvVar("HOME");
		File f = new File(home+"/"+prefsName);
		LinkedHashMap tempcolors = new LinkedHashMap();
		if (f.exists()) {
                   BufferedReader br = new BufferedReader(new FileReader(f));
		   while (br.ready()) {
		      String s = br.readLine();
		      if (s != null && !s.trim().equals("")) {
			StringTokenizer st = new StringTokenizer(s);
			if (st.countTokens() == 2) {
                            String sname = st.nextToken().trim();
                            String rgb = st.nextToken().trim();
                            tempcolors.put(sname, rgb);
                        }
		      }
		   }
		}
		String key, temp;
		for (int j = 0; j < numSpectra; j++) {
		   temp = thePlot.spectra.get(j).colorField.getText();
		   if (!temp.trim().equals("")) {
		      temp = removeWhitespace(temp);
		      if (temp.indexOf(",") == -1) temp = temp.replaceAll(" ", ",");
		   }
		   tempcolors.put(thePlot.spectra.get(j).spectrumName.replaceAll(" ","_"), temp); 
		}
		temp = bgColorField.getText();
		if (!temp.trim().equals("")) {
		   temp = removeWhitespace(temp);
                   if (temp.indexOf(",") == -1) temp = temp.replaceAll(" ", ",");
		}
		tempcolors.put("BG_Color", temp);
                temp = axesColorField.getText();
                if (!temp.trim().equals("")) {
                   temp = removeWhitespace(temp);
                   if (temp.indexOf(",") == -1) temp = temp.replaceAll(" ", ",");
                }
                tempcolors.put("Axes_Color", temp);
		PrintWriter pw = new PrintWriter(new FileOutputStream(f));
		for (Iterator i = tempcolors.keySet().iterator(); i.hasNext(); ) {
		   key = (String)i.next();
		   temp = (String)tempcolors.get(key);
		   pw.println(key+" "+temp);
		}
		pw.close();
		JOptionPane.showMessageDialog(null, "The colors have been saved.", "Colors Saved", JOptionPane.INFORMATION_MESSAGE);
	     } catch (Exception e) {
	      System.err.println("SuperFatboyPlot> "+e.toString());
	     }
	   }
	});

	//initialize vars
        xtickVals = new float[1];
        ytickVals = new float[1];
        xmargin = new float[1];
        ymargin = new float[1];
        position = new float[1];
        xtickNames = new String[1];
        xtickNames[0] = "";
        ytickNames = new String[1];
        ytickNames[0] = "";

	//Ranges
	xMinField = new JTextField(5);
        xMaxField = new JTextField(5);
	yMinField = new JTextField(5);
	yMaxField = new JTextField(5);

	//char size
	charSizeField = new JTextField(2);

	//coord label
	coordLabel = new JLabel("[--, --]");

	//buttons
        addTextButton = new JButton("Add Text");
        addTextButton.addActionListener(new ActionListener() {
          public void actionPerformed(ActionEvent ev) {
            final XYFrame xyFrame = new XYFrame(thePlot);
            thePlot.addxyFrame(xyFrame);
          }
        });

        multiButton = new JButton("Multiplot");
        multiButton.addActionListener(new ActionListener() {
          public void actionPerformed(ActionEvent ev) {
            final JFrame multiFrame = new JFrame("Multiplot");
            multiFrame.setSize(220, 70);
            JPanel multiPanel = new JPanel();
	    SpringLayout multiLayout = new SpringLayout();
	    multiPanel.setLayout(multiLayout);
            multiPanel.setPreferredSize(new Dimension(220, 70));
	    JLabel rowsLabel = new JLabel("Rows:");
            multiPanel.add(rowsLabel);
            multiLayout.putConstraint(SpringLayout.WEST, rowsLabel, 5, SpringLayout.WEST, multiPanel);
            multiLayout.putConstraint(SpringLayout.NORTH, rowsLabel, 5, SpringLayout.NORTH, multiPanel);

            final JTextField rowsField = new JTextField(2);
            multiPanel.add(rowsField);
            multiLayout.putConstraint(SpringLayout.WEST, rowsField, 5, SpringLayout.EAST, rowsLabel);
            multiLayout.putConstraint(SpringLayout.NORTH, rowsField, 5, SpringLayout.NORTH, multiPanel);

	    JLabel colsLabel = new JLabel("Cols");
            multiPanel.add(colsLabel);
            multiLayout.putConstraint(SpringLayout.WEST, colsLabel, 25, SpringLayout.EAST, rowsField);
            multiLayout.putConstraint(SpringLayout.NORTH, colsLabel, 5, SpringLayout.NORTH, multiPanel);

            final JTextField colsField = new JTextField(2);
            multiPanel.add(colsField);
            multiLayout.putConstraint(SpringLayout.WEST, colsField, 5, SpringLayout.EAST, colsLabel);
            multiLayout.putConstraint(SpringLayout.NORTH, colsField, 5, SpringLayout.NORTH, multiPanel);

            JButton apply = new JButton("Apply");
            apply.addActionListener(new ActionListener() {
              public void actionPerformed(ActionEvent ev) {
                int rows = 1, cols = 1;
                if (!rowsField.getText().equals("")) rows = Integer.parseInt(rowsField.getText());
                if (!colsField.getText().equals("")) cols = Integer.parseInt(colsField.getText());
                multi(0, cols, rows);
                multiFrame.dispose();
              }
            });
            multiPanel.add(apply);
            multiLayout.putConstraint(SpringLayout.WEST, apply, 5, SpringLayout.WEST, multiPanel);
            multiLayout.putConstraint(SpringLayout.SOUTH, apply, -10, SpringLayout.SOUTH, multiPanel);
	    multiLayout.putConstraint(SpringLayout.EAST, apply, 100, SpringLayout.WEST, multiPanel);

            JButton cancel = new JButton("Cancel");
            cancel.addActionListener(new ActionListener() {
              public void actionPerformed(ActionEvent ev) {
                multiFrame.dispose();
              }
            });
            multiPanel.add(cancel);
            multiLayout.putConstraint(SpringLayout.EAST, cancel, -5, SpringLayout.EAST, multiPanel);
            multiLayout.putConstraint(SpringLayout.SOUTH, cancel, -10, SpringLayout.SOUTH, multiPanel);

            multiFrame.getContentPane().add(multiPanel);
            multiFrame.pack();
            multiFrame.setVisible(true);
          }
        });

      optsButton = new JButton("Options");
      optsButton.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent ev) {
           final JFrame optsFrame = new JFrame("Options");
           optsFrame.setSize(500, 320);
           JPanel optsPanel = new JPanel();
           SpringLayout optsLayout = new SpringLayout();
           optsPanel.setLayout(optsLayout);
           JLabel xTicksLabel = new JLabel("X ticks:");
           optsPanel.add(xTicksLabel);
           optsLayout.putConstraint(SpringLayout.WEST, xTicksLabel, 5, SpringLayout.WEST, optsPanel);
           optsLayout.putConstraint(SpringLayout.NORTH, xTicksLabel, 5, SpringLayout.NORTH, optsPanel);
           final JTextField optsXTicks = new JTextField(2);
           optsXTicks.setText(""+xticks);
           optsPanel.add(optsXTicks);
           optsLayout.putConstraint(SpringLayout.WEST, optsXTicks, 5, SpringLayout.EAST, xTicksLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, optsXTicks, 5, SpringLayout.NORTH, optsPanel);
           JLabel yTicksLabel = new JLabel("Y ticks:");
           optsPanel.add(yTicksLabel);
           optsLayout.putConstraint(SpringLayout.WEST, yTicksLabel, 10, SpringLayout.EAST, optsXTicks);
           optsLayout.putConstraint(SpringLayout.NORTH, yTicksLabel, 5, SpringLayout.NORTH, optsPanel);
           final JTextField optsYTicks = new JTextField(2);
           optsYTicks.setText(""+yticks);
           optsPanel.add(optsYTicks);
           optsLayout.putConstraint(SpringLayout.WEST, optsYTicks, 5, SpringLayout.EAST, yTicksLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, optsYTicks, 5, SpringLayout.NORTH, optsPanel);
           JLabel xTickIntLabel = new JLabel("X tick interval:");
           optsPanel.add(xTickIntLabel);
           optsLayout.putConstraint(SpringLayout.WEST, xTickIntLabel, 10, SpringLayout.EAST, optsYTicks);
           optsLayout.putConstraint(SpringLayout.NORTH, xTickIntLabel, 5, SpringLayout.NORTH, optsPanel);
           final JTextField optsXTickInt = new JTextField(4);
           optsXTickInt.setText(""+xtickInt);
           optsPanel.add(optsXTickInt);
           optsLayout.putConstraint(SpringLayout.WEST, optsXTickInt, 5, SpringLayout.EAST, xTickIntLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, optsXTickInt, 5, SpringLayout.NORTH, optsPanel);
           JLabel yTickIntLabel = new JLabel("Y tick interval:");
           optsPanel.add(yTickIntLabel);
           optsLayout.putConstraint(SpringLayout.WEST, yTickIntLabel, 10, SpringLayout.EAST, optsXTickInt);
           optsLayout.putConstraint(SpringLayout.NORTH, yTickIntLabel, 5, SpringLayout.NORTH, optsPanel);
           final JTextField optsYTickInt = new JTextField(4);
           optsYTickInt.setText(""+ytickInt);
           optsPanel.add(optsYTickInt);
           optsLayout.putConstraint(SpringLayout.WEST, optsYTickInt, 5, SpringLayout.EAST, yTickIntLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, optsYTickInt, 5, SpringLayout.NORTH, optsPanel);

           JLabel xTickValuesLabel = new JLabel("X tick values:");
           optsPanel.add(xTickValuesLabel);
           optsLayout.putConstraint(SpringLayout.WEST, xTickValuesLabel, 5, SpringLayout.WEST, optsPanel);
           optsLayout.putConstraint(SpringLayout.NORTH, xTickValuesLabel, 10, SpringLayout.SOUTH, optsXTicks);
           final JTextArea optsXTickVals = new JTextArea(4,9);
           for (int j = 0; j < xtickVals.length; j++)
              optsXTickVals.append(""+xtickVals[j]+"\n");
           JScrollPane spOptsXTickVals = new JScrollPane(optsXTickVals);
           optsPanel.add(spOptsXTickVals);
           optsLayout.putConstraint(SpringLayout.WEST, spOptsXTickVals, 5, SpringLayout.EAST, xTickValuesLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, spOptsXTickVals, 10, SpringLayout.SOUTH, optsXTicks);
           JLabel yTickValuesLabel = new JLabel("Y tick values:");
           optsPanel.add(yTickValuesLabel);
           optsLayout.putConstraint(SpringLayout.WEST, yTickValuesLabel, 25, SpringLayout.EAST, spOptsXTickVals);
           optsLayout.putConstraint(SpringLayout.NORTH, yTickValuesLabel, 10, SpringLayout.SOUTH, optsYTicks);
           final JTextArea optsYTickVals = new JTextArea(4,9);
           for (int j = 0; j < ytickVals.length; j++)
              optsYTickVals.append(""+ytickVals[j]+"\n");
           JScrollPane spOptsYTickVals = new JScrollPane(optsYTickVals);
           optsPanel.add(spOptsYTickVals);
           optsLayout.putConstraint(SpringLayout.WEST, spOptsYTickVals, 5, SpringLayout.EAST, yTickValuesLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, spOptsYTickVals, 10, SpringLayout.SOUTH, optsYTicks);

           JLabel xTickNamesLabel = new JLabel("X tick names:");
           optsPanel.add(xTickNamesLabel);
           optsLayout.putConstraint(SpringLayout.WEST, xTickNamesLabel, 5, SpringLayout.WEST, optsPanel);
           optsLayout.putConstraint(SpringLayout.NORTH, xTickNamesLabel, 10, SpringLayout.SOUTH, spOptsXTickVals);
           final JTextArea optsXTickNames = new JTextArea(4,9);
           for (int j = 0; j < xtickNames.length; j++)
              optsXTickNames.append(xtickNames[j]+"\n");
           JScrollPane spOptsXTickNames = new JScrollPane(optsXTickNames);
           optsPanel.add(spOptsXTickNames);
           optsLayout.putConstraint(SpringLayout.WEST, spOptsXTickNames, 5, SpringLayout.EAST, xTickNamesLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, spOptsXTickNames, 10, SpringLayout.SOUTH, spOptsXTickVals);
           JLabel yTickNamesLabel = new JLabel("Y tick names:");
           optsPanel.add(yTickNamesLabel);
           optsLayout.putConstraint(SpringLayout.WEST, yTickNamesLabel, 25, SpringLayout.EAST, spOptsXTickNames);
           optsLayout.putConstraint(SpringLayout.NORTH, yTickNamesLabel, 10, SpringLayout.SOUTH, spOptsYTickVals);
           final JTextArea optsYTickNames = new JTextArea(4,9);
           for (int j = 0; j < ytickNames.length; j++)
              optsYTickNames.append(ytickNames[j]+"\n");
           JScrollPane spOptsYTickNames = new JScrollPane(optsYTickNames);
           optsPanel.add(spOptsYTickNames);
           optsLayout.putConstraint(SpringLayout.WEST, spOptsYTickNames, 5, SpringLayout.EAST, yTickNamesLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, spOptsYTickNames, 10, SpringLayout.SOUTH, spOptsYTickVals);

           JLabel xTickLengthLabel = new JLabel("X tick length:");
           optsPanel.add(xTickLengthLabel);
           optsLayout.putConstraint(SpringLayout.WEST, xTickLengthLabel, 5, SpringLayout.WEST, optsPanel);
           optsLayout.putConstraint(SpringLayout.NORTH, xTickLengthLabel, 10, SpringLayout.SOUTH, spOptsXTickNames);
           final JTextField optsXTickLen = new JTextField(3);
           optsXTickLen.setText(""+xtickLen);
           optsPanel.add(optsXTickLen);
           optsLayout.putConstraint(SpringLayout.WEST, optsXTickLen, 5, SpringLayout.EAST, xTickLengthLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, optsXTickLen, 10, SpringLayout.SOUTH, spOptsXTickNames);
           JLabel yTickLengthLabel = new JLabel("Y tick length:");
           optsPanel.add(yTickLengthLabel);
           optsLayout.putConstraint(SpringLayout.WEST, yTickLengthLabel, 10, SpringLayout.EAST, optsXTickLen);
           optsLayout.putConstraint(SpringLayout.NORTH, yTickLengthLabel, 10, SpringLayout.SOUTH, spOptsXTickNames);
           final JTextField optsYTickLen = new JTextField(3);
           optsYTickLen.setText(""+ytickLen);
           optsPanel.add(optsYTickLen);
           optsLayout.putConstraint(SpringLayout.WEST, optsYTickLen, 5, SpringLayout.EAST, yTickLengthLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, optsYTickLen, 10, SpringLayout.SOUTH, spOptsXTickNames);
           JLabel xMinorLabel = new JLabel("X minor:");
           optsPanel.add(xMinorLabel);
           optsLayout.putConstraint(SpringLayout.WEST, xMinorLabel, 10, SpringLayout.EAST, optsYTickLen);
           optsLayout.putConstraint(SpringLayout.NORTH, xMinorLabel, 10, SpringLayout.SOUTH, spOptsXTickNames);
           final JTextField optsXMinor = new JTextField(2);
           optsXMinor.setText(""+xminor);
           optsPanel.add(optsXMinor);
           optsLayout.putConstraint(SpringLayout.WEST, optsXMinor, 5, SpringLayout.EAST, xMinorLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, optsXMinor, 10, SpringLayout.SOUTH, spOptsXTickNames);
           JLabel yMinorLabel = new JLabel("Y minor:");
           optsPanel.add(yMinorLabel);
           optsLayout.putConstraint(SpringLayout.WEST, yMinorLabel, 10, SpringLayout.EAST, optsXMinor);
           optsLayout.putConstraint(SpringLayout.NORTH, yMinorLabel, 10, SpringLayout.SOUTH, spOptsXTickNames);
           final JTextField optsYMinor = new JTextField(2);
           optsYMinor.setText(""+yminor);
           optsPanel.add(optsYMinor);
           optsLayout.putConstraint(SpringLayout.WEST, optsYMinor, 5, SpringLayout.EAST, yMinorLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, optsYMinor, 10, SpringLayout.SOUTH, spOptsXTickNames);

           JLabel symSizeLabel = new JLabel("Symbol Size:");
           optsPanel.add(symSizeLabel);
           optsLayout.putConstraint(SpringLayout.WEST, symSizeLabel, 5, SpringLayout.WEST, optsPanel);
           optsLayout.putConstraint(SpringLayout.NORTH, symSizeLabel, 10, SpringLayout.SOUTH, optsXTickLen);
           final JTextField optsSymSize = new JTextField(3);
           optsSymSize.setText(""+symsize);
           optsPanel.add(optsSymSize);
           optsLayout.putConstraint(SpringLayout.WEST, optsSymSize, 5, SpringLayout.EAST, symSizeLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, optsSymSize, 10, SpringLayout.SOUTH, optsXTickLen);
           JLabel fontLabel = new JLabel("Font:");
           optsPanel.add(fontLabel);
           optsLayout.putConstraint(SpringLayout.WEST, fontLabel, 10, SpringLayout.EAST, optsSymSize);
           optsLayout.putConstraint(SpringLayout.NORTH, fontLabel, 10, SpringLayout.SOUTH, optsXTickLen);
           final JTextField optsFont = new JTextField(8);
           optsFont.setText(fontName);
           optsPanel.add(optsFont);
           optsLayout.putConstraint(SpringLayout.WEST, optsFont, 5, SpringLayout.EAST, fontLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, optsFont, 10, SpringLayout.SOUTH, optsXTickLen);

           JLabel lMarginLabel = new JLabel("X margin: L");
           optsPanel.add(lMarginLabel);
           optsLayout.putConstraint(SpringLayout.WEST, lMarginLabel, 5, SpringLayout.WEST, optsPanel);
           optsLayout.putConstraint(SpringLayout.NORTH, lMarginLabel, 10, SpringLayout.SOUTH, optsSymSize);
           final JTextField optsXMargin1 = new JTextField(3);
           final JTextField optsXMargin2 = new JTextField(3);
           if (xmargin.length == 2) {
              optsXMargin1.setText(""+xmargin[0]);
              optsXMargin2.setText(""+xmargin[1]);
           }
           optsPanel.add(optsXMargin1);
           optsLayout.putConstraint(SpringLayout.WEST, optsXMargin1, 5, SpringLayout.EAST, lMarginLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, optsXMargin1, 10, SpringLayout.SOUTH, optsSymSize);
           JLabel rMarginLabel = new JLabel("R");
           optsPanel.add(rMarginLabel);
           optsLayout.putConstraint(SpringLayout.WEST, rMarginLabel, 5, SpringLayout.EAST, optsXMargin1);
           optsLayout.putConstraint(SpringLayout.NORTH, rMarginLabel, 10, SpringLayout.SOUTH, optsSymSize);
           optsPanel.add(optsXMargin2);
           optsLayout.putConstraint(SpringLayout.WEST, optsXMargin2, 5, SpringLayout.EAST, rMarginLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, optsXMargin2, 10, SpringLayout.SOUTH, optsSymSize);
           JLabel tMarginLabel = new JLabel("Y margin: T");
           optsPanel.add(tMarginLabel);
           optsLayout.putConstraint(SpringLayout.WEST, tMarginLabel, 30, SpringLayout.EAST, optsXMargin2);
           optsLayout.putConstraint(SpringLayout.NORTH, tMarginLabel, 10, SpringLayout.SOUTH, optsSymSize);
           final JTextField optsYMargin1 = new JTextField(3);
           final JTextField optsYMargin2 = new JTextField(3);
           if (ymargin.length == 2) {
              optsYMargin1.setText(""+ymargin[0]);
              optsYMargin2.setText(""+ymargin[1]);
           }
           optsPanel.add(optsYMargin1);
           optsLayout.putConstraint(SpringLayout.WEST, optsYMargin1, 5, SpringLayout.EAST, tMarginLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, optsYMargin1, 10, SpringLayout.SOUTH, optsSymSize);
           JLabel bMarginLabel = new JLabel("B");
           optsPanel.add(bMarginLabel);
           optsLayout.putConstraint(SpringLayout.WEST, bMarginLabel, 5, SpringLayout.EAST, optsYMargin1);
           optsLayout.putConstraint(SpringLayout.NORTH, bMarginLabel, 10, SpringLayout.SOUTH, optsSymSize);
           optsPanel.add(optsYMargin2);
           optsLayout.putConstraint(SpringLayout.WEST, optsYMargin2, 5, SpringLayout.EAST, bMarginLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, optsYMargin2, 10, SpringLayout.SOUTH, optsSymSize);

           JLabel lPositionLabel = new JLabel("Position: L");
           optsPanel.add(lPositionLabel);
           optsLayout.putConstraint(SpringLayout.WEST, lPositionLabel, 5, SpringLayout.WEST, optsPanel);
           optsLayout.putConstraint(SpringLayout.NORTH, lPositionLabel, 10, SpringLayout.SOUTH, optsXMargin1);
           final JTextField optsPos1 = new JTextField(3);
           final JTextField optsPos2 = new JTextField(3);
           final JTextField optsPos3 = new JTextField(3);
           final JTextField optsPos4 = new JTextField(3);
           if (position.length == 4) {
              optsPos1.setText(""+position[0]);
              optsPos2.setText(""+position[1]);
              optsPos3.setText(""+position[2]);
              optsPos4.setText(""+position[3]);
           }
           optsPanel.add(optsPos1);
           optsLayout.putConstraint(SpringLayout.WEST, optsPos1, 5, SpringLayout.EAST, lPositionLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, optsPos1, 10, SpringLayout.SOUTH, optsXMargin1);
           JLabel tPositionLabel = new JLabel("T");
           optsPanel.add(tPositionLabel);
           optsLayout.putConstraint(SpringLayout.WEST, tPositionLabel, 5, SpringLayout.EAST, optsPos1);
           optsLayout.putConstraint(SpringLayout.NORTH, tPositionLabel, 10, SpringLayout.SOUTH, optsXMargin1);
           optsPanel.add(optsPos2);
           optsLayout.putConstraint(SpringLayout.WEST, optsPos2, 5, SpringLayout.EAST, tPositionLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, optsPos2, 10, SpringLayout.SOUTH, optsXMargin1);
           JLabel rPositionLabel = new JLabel("R");
           optsPanel.add(rPositionLabel);
           optsLayout.putConstraint(SpringLayout.WEST, rPositionLabel, 5, SpringLayout.EAST, optsPos2);
           optsLayout.putConstraint(SpringLayout.NORTH, rPositionLabel, 10, SpringLayout.SOUTH, optsXMargin1);
           optsPanel.add(optsPos3);
           optsLayout.putConstraint(SpringLayout.WEST, optsPos3, 5, SpringLayout.EAST, rPositionLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, optsPos3, 10, SpringLayout.SOUTH, optsXMargin1);
           JLabel bPositionLabel = new JLabel("B");
           optsPanel.add(bPositionLabel);
           optsLayout.putConstraint(SpringLayout.WEST, bPositionLabel, 5, SpringLayout.EAST, optsPos3);
           optsLayout.putConstraint(SpringLayout.NORTH, bPositionLabel, 10, SpringLayout.SOUTH, optsXMargin1);
           optsPanel.add(optsPos4);
           optsLayout.putConstraint(SpringLayout.WEST, optsPos4, 5, SpringLayout.EAST, bPositionLabel);
           optsLayout.putConstraint(SpringLayout.NORTH, optsPos4, 10, SpringLayout.SOUTH, optsXMargin1);

           JButton setPrefs = new JButton("Set Options");
           setPrefs.addActionListener(new ActionListener() {
              public void actionPerformed(ActionEvent ev) {
                String temp;
                String[] temps;
                try {
                   temp = optsXTicks.getText();
                   if (!temp.equals("")) xticks = Integer.parseInt(temp);
                   temp = optsYTicks.getText();
                   if (!temp.equals("")) yticks = Integer.parseInt(temp);
                   temp = optsXTickInt.getText();
                   if (!temp.equals("")) xtickInt = Float.parseFloat(temp);
                   temp = optsYTickInt.getText();
                   if (!temp.equals("")) ytickInt = Float.parseFloat(temp);
                   temp = optsXTickLen.getText();
                   if (!temp.equals("")) xtickLen = Float.parseFloat(temp);
                   temp = optsYTickLen.getText();
                   if (!temp.equals("")) ytickLen = Float.parseFloat(temp);
                   temp = optsXMinor.getText();
                   if (!temp.equals("")) xminor = Integer.parseInt(temp);
                   temp = optsYMinor.getText();
                   if (!temp.equals("")) yminor = Integer.parseInt(temp);
                   temp = optsSymSize.getText();
                   if (!temp.equals("")) symsize = Float.parseFloat(temp);
                   temp = optsFont.getText();
                   if (!temp.equals("")) fontName = temp;
                } catch(Exception e) { System.err.println(e.toString()); }
                temps = optsXTickVals.getText().split("\n");
                if (temps.length != 0)
                   if (!(temps.length == 1 && temps[0].equals(""))) {
                      xtickVals = new float[temps.length];
                      for (int j = 0; j < temps.length; j++)
                        try { xtickVals[j] = Float.parseFloat(temps[j]); }
                        catch(Exception e) { System.err.println(e.toString()); }
                   }
                if (temps.length != 0)
                   temps = optsYTickVals.getText().split("\n");
                   if (!(temps.length == 1 && temps[0].equals(""))) {
                      ytickVals = new float[temps.length];
                      for (int j = 0; j < temps.length; j++)
                        try { ytickVals[j] = Float.parseFloat(temps[j]); }
                        catch(Exception e) { System.err.println(e.toString()); }
                   }

                if (temps.length != 0)
                   temps = optsXTickNames.getText().split("\n");
                   if (!(temps.length == 1 && temps[0].equals(""))) {
                      xtickNames = temps;
                   } else {
                      xtickNames = new String[1];
                      xtickNames[0] = "";
                   }
                if (temps.length != 0)
                   temps = optsYTickNames.getText().split("\n");
                   if (!(temps.length == 1 && temps[0].equals(""))) {
                      ytickNames = temps;
                   } else {
                      ytickNames = new String[1];
                      ytickNames[0] = "";
                   }
                if (!optsXMargin1.getText().equals("") && !optsXMargin2.getText().equals("")) {
                   xmargin = new float[2];
                   try {
                      xmargin[0] = Float.parseFloat(optsXMargin1.getText());
                      xmargin[1] = Float.parseFloat(optsXMargin2.getText());
                   } catch(Exception e) { System.err.println(e.toString()); }
                } else xmargin = new float[1];
                if (!optsYMargin1.getText().equals("") && !optsYMargin2.getText().equals("")) {
                   ymargin = new float[2];
                   try {
                      ymargin[0] = Float.parseFloat(optsYMargin1.getText());
                      ymargin[1] = Float.parseFloat(optsYMargin2.getText());
                   } catch(Exception e) { System.err.println(e.toString()); }
                } else ymargin = new float[1];
                if (!optsPos1.getText().equals("") && !optsPos2.getText().equals("") && !optsPos3.getText().equals("") && !optsPos4.getText().equals("")) {
                   position = new float[4];
                   try {
                      position[0] = Float.parseFloat(optsPos1.getText());
                      position[1] = Float.parseFloat(optsPos2.getText());
                      position[2] = Float.parseFloat(optsPos3.getText());
                      position[3] = Float.parseFloat(optsPos4.getText());
                   } catch(Exception e) { System.err.println(e.toString()); }
                } else position = new float[1];
                optsFrame.dispose();
              }
           });
           JButton cancelPrefs = new JButton("Cancel");
           cancelPrefs.addActionListener(new ActionListener() {
              public void actionPerformed(ActionEvent ev) {
                optsFrame.dispose();
              }
           });
           optsPanel.add(setPrefs);
           optsLayout.putConstraint(SpringLayout.WEST, setPrefs, 135, SpringLayout.WEST, optsPanel);
           optsLayout.putConstraint(SpringLayout.NORTH, setPrefs, 20, SpringLayout.SOUTH, optsPos1);
           optsPanel.add(cancelPrefs);
           optsLayout.putConstraint(SpringLayout.WEST, cancelPrefs, 15, SpringLayout.EAST, setPrefs);
           optsLayout.putConstraint(SpringLayout.NORTH, cancelPrefs, 20, SpringLayout.SOUTH, optsPos1);

           optsPanel.setPreferredSize(new Dimension(480, 340));
           optsFrame.getContentPane().add(optsPanel);
           optsFrame.pack();
           optsFrame.setVisible(true);
        }
      });

        quitButton = new JButton("Quit");
        quitButton.addActionListener(new ActionListener() {
          public void actionPerformed(ActionEvent ev) {
            SuperFatboyPlot.this.dispose();
          }
        });

        URL url = SuperFatboyPlot.class.getResource("gator_small.gif");
        if (url == null) UFButton = new JButton("UF");
        else UFButton = new JButton("UF", new ImageIcon(url));
        UFButton.addActionListener(new ActionListener() {
          public void actionPerformed(ActionEvent ev) {
            float[] x = {0, 1};
            float[] y = {0, 1};
            thePlot.plot(x, y, "*nodata, *color=255, 255, 255");
            xyouts(0.1f, 0.2f, "Go Gators!", "*normal, *color=255, 150, 10, *charsize=48");
            xyouts(0.4f, 0.5f, "Go Gators!", "*normal, *color=0, 0, 255, *charsize=48");
          }
        });

	//lin/log radio buttons
        xLin = new JRadioButton("Linear", true);
        xLog = new JRadioButton("Log");
        xLinLog = new ButtonGroup();
        xLinLog.add(xLin);
        xLinLog.add(xLog);
        xLin.addActionListener(new ActionListener() {
           public void actionPerformed(ActionEvent ev) {
              plotButton.doClick();
           }
        });
        xLog.addActionListener(new ActionListener() {
           public void actionPerformed(ActionEvent ev) {
              plotButton.doClick();
           }
        });

        yLin = new JRadioButton("Linear", true);
        yLog = new JRadioButton("Log");
        yLinLog = new ButtonGroup();
        yLinLog.add(yLin);
        yLinLog.add(yLog);
        yLin.addActionListener(new ActionListener() {
           public void actionPerformed(ActionEvent ev) {
              plotButton.doClick();
           }
        });
        yLog.addActionListener(new ActionListener() {
           public void actionPerformed(ActionEvent ev) {
              plotButton.doClick();
           }
        });

	/* Drag object pulldowns */
        String[] dragObjectOptions = {"Hidden", "H Line", "V Line", "+", "X", "Circle"};
        int[] dragObjectColor = {UFColorCombo.COLOR_SCHEME_RED, UFColorCombo.COLOR_SCHEME_ORANGE, UFColorCombo.COLOR_SCHEME_GREEN, UFColorCombo.COLOR_SCHEME_BLUE};
        dragObjectBox = new UFColorCombo[4];
        for (int j = 0; j < 4; j++) {
            dragObjectBox[j] = new UFColorCombo(dragObjectOptions, dragObjectColor[j]);
            final int objectNum = j;
            dragObjectBox[j].addActionListener(new ActionListener() {
                public void actionPerformed(ActionEvent ev) {
                    thePlot.dragObjectStatus[objectNum]=(String)dragObjectBox[objectNum].getSelectedItem();
                    thePlot.repaint();
                }
            });
        }

	x1l = new JLabel("X1:");
	x2l = new JLabel("X2:");
	dxl = new JLabel("");
	y1l = new JLabel("Y1:");
	y2l = new JLabel("Y2:");
	dyl = new JLabel("");
      }

      public void setupLeftPanel() {
        numSpectra = thePlot.numSpectra;
	leftPanel.removeAll();
        leftLayout = new SpringLayout();
        leftPanel.setLayout(leftLayout);

        /* Spectra / Colors */
        final String[] startingColors = {"0,0,0","255,0,0","0,255,0","0,0,255","180,180,0","180,0,180","0,180,180","128,128,128","0,155,0","255,155,0","155,0,0","0,0,155","255,0,155","0,155,255"};
        for (int j = 0; j < numSpectra; j++) {
           Color tempColor;
           if (colors.containsKey(thePlot.spectra.get(j).spectrumName)) {
              String temp = (String)colors.get(thePlot.spectra.get(j).spectrumName);
              thePlot.spectra.get(j).colorField.setText(temp);
              tempColor = getColor(temp);
           } else {
              thePlot.spectra.get(j).colorField.setText(startingColors[j%startingColors.length]);
              tempColor = getColor(startingColors[j%startingColors.length]);
           }
           if (tempColor != null) thePlot.spectra.get(j).colorField.setForeground(tempColor);
           final JTextField tempColorField = thePlot.spectra.get(j).colorField;
           thePlot.spectra.get(j).colorField.addFocusListener(new FocusListener() {
              public void focusGained(FocusEvent fe) {
              }

              public void focusLost(FocusEvent fe) {
                Color tempColor = getColor(tempColorField.getText());
                if (tempColor != null) {
                   tempColorField.setForeground(tempColor);
                   plotButton.doClick();
                }
              }
           });
        }

	//Top row buttons
	leftPanel.add(addFileButton);
        leftLayout.putConstraint(SpringLayout.WEST, addFileButton, 5, SpringLayout.WEST, leftPanel);
        leftLayout.putConstraint(SpringLayout.NORTH, addFileButton, 5, SpringLayout.NORTH, leftPanel);

	leftPanel.add(renameTabButton);
        leftLayout.putConstraint(SpringLayout.WEST, renameTabButton, 10, SpringLayout.EAST, addFileButton); 
        leftLayout.putConstraint(SpringLayout.NORTH, renameTabButton, 5, SpringLayout.NORTH, leftPanel);

	leftPanel.add(newPanelButton);
        leftLayout.putConstraint(SpringLayout.WEST, newPanelButton, 10, SpringLayout.EAST, renameTabButton); 
        leftLayout.putConstraint(SpringLayout.NORTH, newPanelButton, 5, SpringLayout.NORTH, leftPanel);

	/* Plot Button */
        leftPanel.add(plotButton);
        leftLayout.putConstraint(SpringLayout.WEST, plotButton, 5, SpringLayout.WEST, leftPanel);
        leftLayout.putConstraint(SpringLayout.NORTH, plotButton, 10, SpringLayout.SOUTH, addFileButton); 

	/* Color Chooser */
        leftPanel.add(colorChooser);
        leftLayout.putConstraint(SpringLayout.WEST, colorChooser, 10, SpringLayout.EAST, plotButton);
        leftLayout.putConstraint(SpringLayout.NORTH, colorChooser, 10, SpringLayout.SOUTH, addFileButton); 

	/* Sensors / Colors */
        JLabel fileLabel = new JLabel("Spectra:");
        leftPanel.add(fileLabel);
        leftLayout.putConstraint(SpringLayout.WEST, fileLabel, 5, SpringLayout.WEST, leftPanel);
        leftLayout.putConstraint(SpringLayout.NORTH, fileLabel, 10, SpringLayout.SOUTH, plotButton);
        int maxLength = 0, nmax=0;
        boolean[] hasLabel = new boolean[numSpectra];
        JLabel tempLabel = null;
        for (int j = 0; j < numSpectra; j++) {
           final int myJ = j;
           hasLabel[j] = false;
	   for (int l = 0; l < thePlot.rssContainers.size(); l++) {
	      UFRSSPlot.RSSContainer theContainer = thePlot.rssContainers.get(l);
	      if (myJ == theContainer.position) {
                tempLabel = theContainer.fileLabel; 
                leftPanel.add(theContainer.plotAllCheckbox);
                leftLayout.putConstraint(SpringLayout.WEST, theContainer.plotAllCheckbox, 5, SpringLayout.WEST, leftPanel);
                if (j == 0) {
                   leftLayout.putConstraint(SpringLayout.NORTH, theContainer.plotAllCheckbox, 10, SpringLayout.SOUTH, fileLabel);
                } else {
                   leftLayout.putConstraint(SpringLayout.NORTH, theContainer.plotAllCheckbox, 10, SpringLayout.SOUTH, thePlot.spectra.get(j-1).plotCheckbox);
                }
                leftPanel.add(tempLabel);
                leftLayout.putConstraint(SpringLayout.WEST, tempLabel, 5, SpringLayout.EAST, theContainer.plotAllCheckbox);
                if (j == 0) {
                   leftLayout.putConstraint(SpringLayout.NORTH, tempLabel, 10, SpringLayout.SOUTH, fileLabel);
                } else {
                   leftLayout.putConstraint(SpringLayout.NORTH, tempLabel, 10, SpringLayout.SOUTH, thePlot.spectra.get(j-1).plotCheckbox);
                }
		leftPanel.add(theContainer.deleteButton);
                leftLayout.putConstraint(SpringLayout.WEST, theContainer.deleteButton, 10, SpringLayout.EAST, tempLabel);
                if (j == 0) {
                   leftLayout.putConstraint(SpringLayout.NORTH, theContainer.deleteButton, 10, SpringLayout.SOUTH, fileLabel);
                } else {
                   leftLayout.putConstraint(SpringLayout.NORTH, theContainer.deleteButton, 10, SpringLayout.SOUTH, thePlot.spectra.get(j-1).plotCheckbox);
                }
                hasLabel[j] = true;
              }
           }
           if (thePlot.spectra.get(j).spectrumName.length() > maxLength) {
              maxLength = thePlot.spectra.get(j).spectrumName.length();
              nmax = j;
           }
           leftPanel.add(thePlot.spectra.get(j).plotCheckbox);
           leftLayout.putConstraint(SpringLayout.WEST, thePlot.spectra.get(j).plotCheckbox, 5, SpringLayout.WEST, leftPanel);
           if (hasLabel[j]) {
              leftLayout.putConstraint(SpringLayout.NORTH, thePlot.spectra.get(j).plotCheckbox, 15, SpringLayout.SOUTH, tempLabel);
           } else if (j == 0) {
              leftLayout.putConstraint(SpringLayout.NORTH, thePlot.spectra.get(j).plotCheckbox, 0, SpringLayout.SOUTH, fileLabel);
           } else {
              leftLayout.putConstraint(SpringLayout.NORTH, thePlot.spectra.get(j).plotCheckbox, 0, SpringLayout.SOUTH, thePlot.spectra.get(j-1).plotCheckbox);
           }
        }

        JLabel bgLabel = new JLabel("BG Color:");
        leftPanel.add(bgLabel);
        leftLayout.putConstraint(SpringLayout.WEST, bgLabel, 5, SpringLayout.WEST, leftPanel);
        if (numSpectra == 0) {
          leftLayout.putConstraint(SpringLayout.NORTH, bgLabel, 10, SpringLayout.SOUTH, fileLabel);
        } else {
          leftLayout.putConstraint(SpringLayout.NORTH, bgLabel, 10, SpringLayout.SOUTH, thePlot.spectra.get(numSpectra-1).plotCheckbox);
        }

        JLabel axesColorLabel = new JLabel("Axes Color:");
        leftPanel.add(axesColorLabel);
        leftLayout.putConstraint(SpringLayout.WEST, axesColorLabel, 5, SpringLayout.WEST, leftPanel);
        leftLayout.putConstraint(SpringLayout.NORTH, axesColorLabel, 10, SpringLayout.SOUTH, bgLabel);

        leftPanel.add(bgColorField);
	if (numSpectra == 0) {
          leftLayout.putConstraint(SpringLayout.WEST, bgColorField, 10, SpringLayout.EAST, axesColorLabel);
          leftLayout.putConstraint(SpringLayout.NORTH, bgColorField, 10, SpringLayout.SOUTH, fileLabel); 
	} else {
          leftLayout.putConstraint(SpringLayout.WEST, bgColorField, 10, SpringLayout.EAST, thePlot.spectra.get(nmax).plotCheckbox);
          leftLayout.putConstraint(SpringLayout.NORTH, bgColorField, 10, SpringLayout.SOUTH, thePlot.spectra.get(numSpectra-1).plotCheckbox);
	}

        leftPanel.add(axesColorField);
	if (numSpectra == 0) {
          leftLayout.putConstraint(SpringLayout.WEST, axesColorField, 10, SpringLayout.EAST, axesColorLabel); 
        } else { 
          leftLayout.putConstraint(SpringLayout.WEST, axesColorField, 10, SpringLayout.EAST, thePlot.spectra.get(nmax).plotCheckbox);
	}
        leftLayout.putConstraint(SpringLayout.NORTH, axesColorField, 10, SpringLayout.SOUTH, bgLabel);

        JLabel colorLabel = new JLabel("Color (R,G,B):");
        leftPanel.add(colorLabel);
	if (numSpectra == 0) {
          leftLayout.putConstraint(SpringLayout.WEST, colorLabel, 10, SpringLayout.EAST, axesColorLabel); 
	} else {
          leftLayout.putConstraint(SpringLayout.WEST, colorLabel, 10, SpringLayout.EAST, thePlot.spectra.get(nmax).plotCheckbox);
	}
        leftLayout.putConstraint(SpringLayout.NORTH, colorLabel, 10, SpringLayout.SOUTH, plotButton);
        for (int j = 0; j < numSpectra; j++) {
          leftPanel.add(thePlot.spectra.get(j).colorField);
          leftLayout.putConstraint(SpringLayout.WEST, thePlot.spectra.get(j).colorField, 10, SpringLayout.EAST, thePlot.spectra.get(nmax).plotCheckbox);
          if (hasLabel[j]) {
            leftLayout.putConstraint(SpringLayout.NORTH, thePlot.spectra.get(j).colorField, 4, SpringLayout.NORTH, thePlot.spectra.get(j).plotCheckbox);
          } else if (j == 0) {
            leftLayout.putConstraint(SpringLayout.NORTH, thePlot.spectra.get(j).colorField, 4, SpringLayout.SOUTH, fileLabel);
          } else {
            leftLayout.putConstraint(SpringLayout.NORTH, thePlot.spectra.get(j).colorField, 4, SpringLayout.SOUTH, thePlot.spectra.get(j-1).plotCheckbox);
          }
        }

	//Plotting symbols
	if (numSpectra > 0) {
	  JLabel symLabel = new JLabel("PSYM:");
          leftPanel.add(symLabel);
          leftLayout.putConstraint(SpringLayout.WEST, symLabel, 10, SpringLayout.EAST, thePlot.spectra.get(nmax).colorField);
          leftLayout.putConstraint(SpringLayout.NORTH, symLabel, 10, SpringLayout.SOUTH, plotButton);
          for (int j = 0; j < numSpectra; j++) {
	    leftPanel.add(thePlot.spectra.get(j).symField);
            leftLayout.putConstraint(SpringLayout.WEST, thePlot.spectra.get(j).symField, 10, SpringLayout.EAST, thePlot.spectra.get(nmax).colorField);
            if (hasLabel[j]) {
              leftLayout.putConstraint(SpringLayout.NORTH, thePlot.spectra.get(j).symField, 4, SpringLayout.NORTH, thePlot.spectra.get(j).plotCheckbox);
            } else if (j == 0) {
              leftLayout.putConstraint(SpringLayout.NORTH, thePlot.spectra.get(j).symField, 4, SpringLayout.SOUTH, fileLabel);
            } else {
              leftLayout.putConstraint(SpringLayout.NORTH, thePlot.spectra.get(j).symField, 4, SpringLayout.SOUTH, thePlot.spectra.get(j-1).plotCheckbox);
            }
          }
	}

	/* Save Color Button */
        leftPanel.add(saveColorButton);
        leftLayout.putConstraint(SpringLayout.WEST, saveColorButton, 5, SpringLayout.WEST, leftPanel);
        leftLayout.putConstraint(SpringLayout.NORTH, saveColorButton, 10, SpringLayout.SOUTH, axesColorField);
        if (numSpectra > 10) {
           leftLayout.putConstraint(SpringLayout.SOUTH, leftPanel, 10, SpringLayout.SOUTH, saveColorButton);
        }

	leftPanel.revalidate();
	leftPanel.repaint();
      }

      public void drawComponents() {
        numSpectra = thePlot.numSpectra;
        setLayout(new BorderLayout());
        setPreferredSize(new Dimension(964, 640));

	//setup bottom panel
        JPanel bottomPanel = new JPanel();
        SpringLayout bottomLayout = new SpringLayout();
        bottomPanel.setLayout(bottomLayout);
        bottomPanel.setPreferredSize(new Dimension(640, 128));

	//title fields
        titleField = new JTextField(10);
        xtitleField = new JTextField(9);
        ytitleField = new JTextField(9);

	JLabel titleLabel = new JLabel("Title:");
        bottomPanel.add(titleLabel);
        bottomLayout.putConstraint(SpringLayout.WEST, titleLabel, 5, SpringLayout.WEST, bottomPanel);
        bottomLayout.putConstraint(SpringLayout.NORTH, titleLabel, 10, SpringLayout.NORTH, bottomPanel);
        bottomPanel.add(titleField);
        bottomLayout.putConstraint(SpringLayout.WEST, titleField, 5, SpringLayout.EAST, titleLabel);
        bottomLayout.putConstraint(SpringLayout.NORTH, titleField, 8, SpringLayout.NORTH, bottomPanel); 

        JLabel xtitleLabel = new JLabel("x-title:");
        bottomPanel.add(xtitleLabel);
        bottomLayout.putConstraint(SpringLayout.WEST, xtitleLabel, 5, SpringLayout.WEST, bottomPanel);
        bottomLayout.putConstraint(SpringLayout.NORTH, xtitleLabel, 10, SpringLayout.SOUTH, titleLabel);
        bottomPanel.add(xtitleField);
        bottomLayout.putConstraint(SpringLayout.WEST, xtitleField, 5, SpringLayout.EAST, xtitleLabel);
        bottomLayout.putConstraint(SpringLayout.NORTH, xtitleField, 8, SpringLayout.SOUTH, titleLabel);

        JLabel ytitleLabel = new JLabel("y-title:");
        bottomPanel.add(ytitleLabel);
        bottomLayout.putConstraint(SpringLayout.WEST, ytitleLabel, 5, SpringLayout.WEST, bottomPanel);
        bottomLayout.putConstraint(SpringLayout.NORTH, ytitleLabel, 10, SpringLayout.SOUTH, xtitleLabel);
        bottomPanel.add(ytitleField);
        bottomLayout.putConstraint(SpringLayout.WEST, ytitleField, 5, SpringLayout.EAST, ytitleLabel);
        bottomLayout.putConstraint(SpringLayout.NORTH, ytitleField, 8, SpringLayout.SOUTH, xtitleLabel);

        JLabel charSizeLabel = new JLabel("CharSize:");
        bottomPanel.add(charSizeLabel);
        bottomLayout.putConstraint(SpringLayout.WEST, charSizeLabel, 25, SpringLayout.EAST, titleField);
        bottomLayout.putConstraint(SpringLayout.NORTH, charSizeLabel, 10, SpringLayout.NORTH, bottomPanel);
        bottomPanel.add(charSizeField);
        bottomLayout.putConstraint(SpringLayout.WEST, charSizeField, 5, SpringLayout.EAST, charSizeLabel);
        bottomLayout.putConstraint(SpringLayout.NORTH, charSizeField, 8, SpringLayout.NORTH, bottomPanel);

	//ranges
        JLabel xRangeLabel = new JLabel("x-range:");
        bottomPanel.add(xRangeLabel);
        bottomLayout.putConstraint(SpringLayout.WEST, xRangeLabel, 15, SpringLayout.EAST, charSizeField); 
        bottomLayout.putConstraint(SpringLayout.NORTH, xRangeLabel, 10, SpringLayout.NORTH, bottomPanel);
        bottomPanel.add(xMinField);
        bottomLayout.putConstraint(SpringLayout.WEST, xMinField, 5, SpringLayout.EAST, xRangeLabel);
        bottomLayout.putConstraint(SpringLayout.NORTH, xMinField, 8, SpringLayout.NORTH, bottomPanel);
        JLabel xToLabel = new JLabel("to");
        bottomPanel.add(xToLabel);
        bottomLayout.putConstraint(SpringLayout.WEST, xToLabel, 5, SpringLayout.EAST, xMinField);
        bottomLayout.putConstraint(SpringLayout.NORTH, xToLabel, 10, SpringLayout.NORTH, bottomPanel);
        bottomPanel.add(xMaxField);
        bottomLayout.putConstraint(SpringLayout.WEST, xMaxField, 5, SpringLayout.EAST, xToLabel);
        bottomLayout.putConstraint(SpringLayout.NORTH, xMaxField, 8, SpringLayout.NORTH, bottomPanel);

        JLabel yRangeLabel = new JLabel("y-range:");
        bottomPanel.add(yRangeLabel);
        bottomLayout.putConstraint(SpringLayout.WEST, yRangeLabel, 15, SpringLayout.EAST, xMaxField);
        bottomLayout.putConstraint(SpringLayout.NORTH, yRangeLabel, 10, SpringLayout.NORTH, bottomPanel);
        bottomPanel.add(yMinField);
        bottomLayout.putConstraint(SpringLayout.WEST, yMinField, 5, SpringLayout.EAST, yRangeLabel);
        bottomLayout.putConstraint(SpringLayout.NORTH, yMinField, 8, SpringLayout.NORTH, bottomPanel);
        JLabel yToLabel = new JLabel("to");
        bottomPanel.add(yToLabel);
        bottomLayout.putConstraint(SpringLayout.WEST, yToLabel, 5, SpringLayout.EAST, yMinField);
        bottomLayout.putConstraint(SpringLayout.NORTH, yToLabel, 10, SpringLayout.NORTH, bottomPanel);
        bottomPanel.add(yMaxField);
        bottomLayout.putConstraint(SpringLayout.WEST, yMaxField, 5, SpringLayout.EAST, yToLabel);
        bottomLayout.putConstraint(SpringLayout.NORTH, yMaxField, 8, SpringLayout.NORTH, bottomPanel);

	//coord label
	bottomPanel.add(coordLabel);
	bottomLayout.putConstraint(SpringLayout.WEST, coordLabel, 10, SpringLayout.EAST, yMaxField);
        bottomLayout.putConstraint(SpringLayout.NORTH, coordLabel, 10, SpringLayout.NORTH, bottomPanel);

	//Lin/log
        JLabel xLogLabel = new JLabel("X-axis:");
        bottomPanel.add(xLogLabel);
        bottomLayout.putConstraint(SpringLayout.WEST, xLogLabel, 25, SpringLayout.EAST, xtitleField); 
        bottomLayout.putConstraint(SpringLayout.NORTH, xLogLabel, 15, SpringLayout.SOUTH, charSizeLabel);
        bottomPanel.add(xLin);
        bottomLayout.putConstraint(SpringLayout.WEST, xLin, 5, SpringLayout.EAST, xLogLabel);
        bottomLayout.putConstraint(SpringLayout.NORTH, xLin, 12, SpringLayout.SOUTH, charSizeLabel); 
        bottomPanel.add(xLog);
        bottomLayout.putConstraint(SpringLayout.WEST, xLog, 5, SpringLayout.EAST, xLin);
        bottomLayout.putConstraint(SpringLayout.NORTH, xLog, 12, SpringLayout.SOUTH, charSizeLabel);
        JLabel yLogLabel = new JLabel("Y-axis:");
        bottomPanel.add(yLogLabel);
        bottomLayout.putConstraint(SpringLayout.WEST, yLogLabel, 20, SpringLayout.EAST, xLog);
        bottomLayout.putConstraint(SpringLayout.NORTH, yLogLabel, 15, SpringLayout.SOUTH, charSizeLabel); 
        bottomPanel.add(yLin);
        bottomLayout.putConstraint(SpringLayout.WEST, yLin, 5, SpringLayout.EAST, yLogLabel);
        bottomLayout.putConstraint(SpringLayout.NORTH, yLin, 12, SpringLayout.SOUTH, charSizeLabel); 
        bottomPanel.add(yLog);
        bottomLayout.putConstraint(SpringLayout.WEST, yLog, 5, SpringLayout.EAST, yLin);
        bottomLayout.putConstraint(SpringLayout.NORTH, yLog, 12, SpringLayout.SOUTH, charSizeLabel);

	//buttons - middle row
        bottomPanel.add(addTextButton);
        bottomLayout.putConstraint(SpringLayout.WEST, addTextButton, 30, SpringLayout.EAST, yLog);
        bottomLayout.putConstraint(SpringLayout.NORTH, addTextButton, 10, SpringLayout.SOUTH, charSizeLabel);

        bottomPanel.add(multiButton);
        bottomLayout.putConstraint(SpringLayout.WEST, multiButton, 20, SpringLayout.EAST, addTextButton);
        bottomLayout.putConstraint(SpringLayout.NORTH, multiButton, 10, SpringLayout.SOUTH, charSizeLabel);

        bottomPanel.add(optsButton);
        bottomLayout.putConstraint(SpringLayout.WEST, optsButton, 20, SpringLayout.EAST, multiButton);
        bottomLayout.putConstraint(SpringLayout.NORTH, optsButton, 10, SpringLayout.SOUTH, charSizeLabel);

	//buttons - bottom row
        bottomPanel.add(quitButton);
        bottomLayout.putConstraint(SpringLayout.WEST, quitButton, 10, SpringLayout.WEST, bottomPanel);
        bottomLayout.putConstraint(SpringLayout.NORTH, quitButton, 10, SpringLayout.SOUTH, ytitleLabel);

        bottomPanel.add(UFButton);
        bottomLayout.putConstraint(SpringLayout.EAST, UFButton, -10, SpringLayout.EAST, bottomPanel);
        bottomLayout.putConstraint(SpringLayout.NORTH, UFButton, 80, SpringLayout.NORTH, bottomPanel);

	/* 2nd plot button */
        JButton plotButton2 = new JButton("Plot");
        bottomPanel.add(plotButton2);
        bottomLayout.putConstraint(SpringLayout.EAST, plotButton2, -10, SpringLayout.EAST, bottomPanel); 
        bottomLayout.putConstraint(SpringLayout.NORTH, plotButton2, 5, SpringLayout.NORTH, bottomPanel);
        plotButton2.addActionListener(new ActionListener() {
           public void actionPerformed(ActionEvent ev) {
              plotButton.doClick();
           }
        });

	/* Add drag objects */
        for (int j = 0; j < 4; j++) {
            bottomPanel.add(dragObjectBox[j]);
            if (j == 0) {
                bottomLayout.putConstraint(SpringLayout.WEST, dragObjectBox[j], 25, SpringLayout.EAST, ytitleField);
            } else {
                bottomLayout.putConstraint(SpringLayout.WEST, dragObjectBox[j], 15, SpringLayout.EAST, dragObjectBox[j-1]);
            }
            bottomLayout.putConstraint(SpringLayout.NORTH, dragObjectBox[j], 20, SpringLayout.SOUTH, xtitleField);
        }


	/* Add drag object xy labels */
        bottomPanel.add(x1l);
        bottomLayout.putConstraint(SpringLayout.WEST, x1l, 15, SpringLayout.EAST, dragObjectBox[3]);
        bottomLayout.putConstraint(SpringLayout.NORTH, x1l, 15, SpringLayout.SOUTH, xtitleField);
        bottomPanel.add(x2l);
        bottomLayout.putConstraint(SpringLayout.WEST, x2l, 10, SpringLayout.EAST, x1l);
        bottomLayout.putConstraint(SpringLayout.NORTH, x2l, 15, SpringLayout.SOUTH, xtitleField);
        bottomPanel.add(dxl);
        bottomLayout.putConstraint(SpringLayout.WEST, dxl, 10, SpringLayout.EAST, x2l);
        bottomLayout.putConstraint(SpringLayout.NORTH, dxl, 15, SpringLayout.SOUTH, xtitleField);
        bottomPanel.add(y1l);
        bottomLayout.putConstraint(SpringLayout.WEST, y1l, 15, SpringLayout.EAST, dragObjectBox[3]);
        bottomLayout.putConstraint(SpringLayout.NORTH, y1l, 10, SpringLayout.SOUTH, x1l);
        bottomPanel.add(y2l);
        bottomLayout.putConstraint(SpringLayout.WEST, y2l, 10, SpringLayout.EAST, y1l);
        bottomLayout.putConstraint(SpringLayout.NORTH, y2l, 10, SpringLayout.SOUTH, x1l);
        bottomPanel.add(dyl);
        bottomLayout.putConstraint(SpringLayout.WEST, dyl, 10, SpringLayout.EAST, y2l);
        bottomLayout.putConstraint(SpringLayout.NORTH, dyl, 10, SpringLayout.SOUTH, x1l);

        JScrollPane sp = new JScrollPane(leftPanel);
        sp.setPreferredSize(new Dimension(324, 512));
        add(thePlot, BorderLayout.CENTER);
        add(sp, BorderLayout.WEST);
        add(bottomPanel, BorderLayout.SOUTH);
      }

      public void startPlot() {
        //pass initial options to plot
        String opts = readOpts();
        String[] oPlotOpts = readOplotOpts(numSpectra);
        thePlot.updatePlotOpts(opts, oPlotOpts);
      }

      public String removeWhitespace(String s) {
	while (s.indexOf("\t") != -1) s = s.replaceAll("\t"," ");
	while (s.indexOf("  ") != -1) {
	   s = s.replaceAll("  "," ");
	}
	s = s.trim();
	return s;
      }

      public String readOpts() {
	String s = "";
        String temp;

	temp = titleField.getText();
        if (!temp.equals("")) {
          s+="*title="+temp+", ";
          SuperFatboyPlot.this.setTitle("SuperFatboyPlot: " + temp);
        }

        temp = xtitleField.getText();
        if (!temp.equals("")) s+="*xtitle="+temp+", ";
        temp = ytitleField.getText();
        if (!temp.equals("")) s+="*ytitle="+temp+", ";
        temp = charSizeField.getText();
        if (!temp.trim().equals("")) s+="*charsize="+temp+", ";

        temp = xMinField.getText();
        if (!temp.trim().equals("")) s+="*xminval="+temp+", ";
        temp = xMaxField.getText();
        if (!temp.trim().equals("")) s+="*xmaxval="+temp+", ";
        temp = yMinField.getText();
        if (!temp.trim().equals("")) s+="*yminval="+temp+", ";
        temp = yMaxField.getText();
        if (!temp.trim().equals("")) s+="*ymaxval="+temp+", ";
	temp = bgColorField.getText();
	if (!temp.trim().equals("")) {
	   temp = removeWhitespace(temp);
	   if (temp.indexOf(",") == -1) temp = temp.replaceAll(" ", ",");
	   s+="*background="+temp+", ";
	   int colorLen = temp.split(",").length;
	   if (colorLen < 3) {
	      for (int l = 0; l < 3-colorLen; l++) s+=temp.substring(temp.lastIndexOf(",")+1)+", ";
	   }
	}
        temp = axesColorField.getText();
        if (!temp.trim().equals("")) {
           temp = removeWhitespace(temp);
           if (temp.indexOf(",") == -1) temp = temp.replaceAll(" ", ",");
           s+="*axescolor="+temp+", ";
           int colorLen = temp.split(",").length;
           if (colorLen < 3) {
              for (int l = 0; l < 3-colorLen; l++) s+=temp.substring(temp.lastIndexOf(",")+1)+", ";
           }
        }

        if (xLog.isSelected()) s+="*xlog, ";
        if (yLog.isSelected()) s+="*ylog, ";
        if (xLin.isSelected()) s+="*xlinear, ";
        if (yLin.isSelected()) s+="*ylinear, ";

        if (xticks != 0) s+="*xticks="+xticks+", ";
        if (yticks != 0) s+="*yticks="+yticks+", ";
        if (xminor != 0) s+="*xminor="+xminor+", ";
        if (yminor != 0) s+="*yminor="+yminor+", ";
        if (xtickInt != 0) s+="*xtickinterval="+xtickInt+", ";
        if (ytickInt != 0) s+="*ytickinterval="+ytickInt+", ";
        if (xtickLen != 0) s+="*xticklen="+xtickLen+", ";
        if (ytickLen != 0) s+="*yticklen="+ytickLen+", ";
        if (!fontName.equals("")) s+="*font="+fontName+", ";
        if (xtickVals.length != 0) {
          if (!(xtickVals.length == 1 && xtickVals[0] == 0)) {
            s+="*xtickv=[";
            for (int j = 0; j < xtickVals.length-1; j++) s+=xtickVals[j]+",";
            s+=xtickVals[xtickVals.length-1]+"], ";
          }
	}
        if (ytickVals.length != 0) {
          if (!(ytickVals.length == 1 && ytickVals[0] == 0)) {
            s+="*ytickv=[";
            for (int j = 0; j < ytickVals.length-1; j++) s+=ytickVals[j]+",";
            s+=ytickVals[ytickVals.length-1]+"], ";
          }
        }
        if (xtickNames.length != 0) {
          if (!(xtickNames.length == 1 && xtickNames[0].trim().equals(""))) {
            s+="*xtickname=[";
            for (int j = 0; j < xtickNames.length-1; j++) s+=xtickNames[j]+",";
            s+=xtickNames[xtickNames.length-1]+"], ";
          }
	}
        if (ytickNames.length != 0) {
          if (!(ytickNames.length == 1 && ytickNames[0].trim().equals(""))) {
            s+="*ytickname=[";
            for (int j = 0; j < ytickNames.length-1; j++) s+=ytickNames[j]+",";
            s+=ytickNames[ytickNames.length-1]+"], ";
          }
	}
        if (xmargin.length == 2) s+="*xmargin=["+xmargin[0]+","+xmargin[1]+"], ";
        if (ymargin.length == 2) s+="*ymargin=["+ymargin[0]+","+ymargin[1]+"], ";
        if (position.length == 4) {
          s+="*position=["+position[0]+","+position[1]+","+position[2]+","+position[3]+"], ";
        }

	return s;
      }

      public String[] readOplotOpts(int numSpectra) {
        String temp;
        String[] oPlotOpts = new String[numSpectra];
        for (int j = 0; j < numSpectra; j++) {
           oPlotOpts[j] = "";
           temp = thePlot.spectra.get(j).colorField.getText();
           if (!temp.trim().equals("")) {
              temp = removeWhitespace(temp);
              if (temp.indexOf(",") == -1) temp = temp.replaceAll(" ", ",");
              oPlotOpts[j]+="*color="+temp+", ";
              int colorLen = temp.split(",").length;
              if (colorLen < 3) {
                for (int l = 0; l < 3-colorLen; l++) oPlotOpts[j]+=temp.substring(temp.lastIndexOf(",")+1)+", ";
              }
           }
	   temp = thePlot.spectra.get(j).symField.getText();
           if (!temp.trim().equals("")) {
              temp = removeWhitespace(temp);
	      oPlotOpts[j]+="*psym="+temp+", ";
              if (symsize != 0) oPlotOpts[j]+="*symsize="+symsize+", ";
           }
           if (xLog.isSelected()) oPlotOpts[j] += "*xlog, ";
           if (yLog.isSelected()) oPlotOpts[j] += "*ylog, ";
        }
        return oPlotOpts;
      }

      public Color getColor(String temp) {
	if (!temp.trim().equals("")) {
	   temp = removeWhitespace(temp);
	}
	if (temp.indexOf(",") == -1) temp = temp.replaceAll(" ", ",");
	String[] temprgb = temp.split(",");
	if (temprgb.length < 3) return Color.BLACK;
	int r = Integer.parseInt(temprgb[0].trim());
        int g = Integer.parseInt(temprgb[1].trim());
        int b = Integer.parseInt(temprgb[2].trim());
	return new Color(r,g,b);
      }

      public void plot() {
	thePlot.updatePlot();
      }

      public void xyouts(float xc, float yc, String text, String s) {
        thePlot.xyouts(xc, yc, text, s);
      }

      public void multi(int curr, int col, int row) {
        thePlot.multi(curr, col, row);
	if (col*row == 1) multiMode = false; else multiMode = true;
      }

      public void updateCoordLabel(String s) {
	coordLabel.setText(s);
      }

      public void updateSize(int x, int y) {
	SuperFatboyPlot.this.setSize(x, y);
      }

    }

    public static void main(String[] args) {
	new SuperFatboyPlot(args);
    }
}
