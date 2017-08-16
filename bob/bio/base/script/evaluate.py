#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <manuel.guenther@idiap.ch>
# Tue Jul 2 14:52:49 CEST 2013

from __future__ import print_function

"""This script evaluates the given score files and computes EER, HTER.
It also is able to plot CMC and ROC curves."""

import bob.measure

import argparse
import numpy, math
import os
import scipy.stats

# matplotlib stuff
import matplotlib; matplotlib.use('pdf') #avoids TkInter threaded start
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

from scipy.stats import norm
import statsmodels.api as sm
import scipy.stats


# enable LaTeX interpreter
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
matplotlib.rc('lines', linewidth = 4)
# increase the default font size
matplotlib.rc('font', size=18)

import bob.core
logger = bob.core.log.setup("bob.bio.base")


def command_line_arguments(command_line_parameters):
  """Parse the program options"""

  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-d', '--dev-files', required=True, nargs='+', help = "A list of score files of the development set.")
  parser.add_argument('-e', '--eval-files', nargs='+', help = "A list of score files of the evaluation set; if given it must be the same number of files as the --dev-files.")

  parser.add_argument('-s', '--directory', default = '.', help = "A directory, where to find the --dev-files and the --eval-files")

  parser.add_argument('-c', '--criterion', choices = ('EER', 'HTER', 'FAR'), help = "If given, the threshold of the development set will be computed with this criterion.")
  parser.add_argument('-f', '--far-value', type=float, default=0.1, help = "The FAR value in %% for which to evaluate (only for --criterion FAR)")
  parser.add_argument('-x', '--cllr', action = 'store_true', help = "If given, Cllr and minCllr will be computed.")
  parser.add_argument('-m', '--mindcf', action = 'store_true', help = "If given, minDCF will be computed.")
  parser.add_argument('--cost', default=0.99,  help='Cost for FAR in minDCF')
  parser.add_argument('-r', '--rr', action = 'store_true', help = "If given, the Recognition Rate will be computed.")
  parser.add_argument('-t', '--thresholds', type=float, nargs='+', help = "If given, the Recognition Rate will incorporate an Open Set handling, rejecting all scores that are below the given threshold; when multiple thresholds are given, they are applied in the same order as the --dev-files.")
  parser.add_argument('-l', '--legends', nargs='+', help = "A list of legend strings used for ROC, CMC and DET plots; if given, must be the same number than --dev-files.")
  parser.add_argument('-F', '--legend-font-size', type=int, default=18, help = "Set the font size of the legends.")
  parser.add_argument('-P', '--legend-position', type=int, help = "Set the font size of the legends.")
  parser.add_argument('-T', '--title', nargs = '+', help = "Overwrite the default title of the plot for development (and evaluation) set")
  parser.add_argument('-R', '--roc', help = "If given, ROC curves will be plotted into the given pdf file.")
  parser.add_argument('-D', '--det', help = "If given, DET curves will be plotted into the given pdf file.")
  parser.add_argument('-C', '--cmc', help = "If given, CMC curves will be plotted into the given pdf file.")
  parser.add_argument('-E', '--epc', help = "If given, EPC curves will be plotted into the given pdf file. For this plot --eval-files is mandatory.")
  parser.add_argument('-GID', '--GIdistr', help = "If given, the genuine and impostor score distributions will be plotted into the given pdf file.")
  parser.add_argument('--parser', default = '4column', choices = ('4column', '5column'), help="The style of the resulting score files. The default fits to the usual output of score files.")

  # add verbose option
  bob.core.log.add_command_line_option(parser)

  # parse arguments
  args = parser.parse_args(command_line_parameters)

  # set verbosity level
  bob.core.log.set_verbosity_level(logger, args.verbose)


  # some sanity checks:
  if args.eval_files is not None and len(args.dev_files) != len(args.eval_files):
    logger.error("The number of --dev-files (%d) and --eval-files (%d) are not identical", len(args.dev_files), len(args.eval_files))

  # update legends when they are not specified on command line
  if args.legends is None:
    args.legends = [f.replace('_', '-') for f in args.dev_files]
    logger.warn("Legends are not specified; using legends estimated from --dev-files: %s", args.legends)

  # check that the legends have the same length as the dev-files
  if len(args.dev_files) != len(args.legends):
    logger.error("The number of --dev-files (%d) and --legends (%d) are not identical", len(args.dev_files), len(args.legends))

  if args.thresholds is not None:
    if len(args.thresholds) == 1:
      args.thresholds = args.thresholds * len(args.dev_files)
    elif len(args.thresholds) != len(args.dev_files):
      logger.error("If given, the number of --thresholds imust be either 1, or the same as --dev-files (%d), but it is %d", len(args.dev_files), len(args.thresholds))
  else:
    args.thresholds = [None] * len(args.dev_files)

  if args.title is not None:
    if args.eval_files is None and len(args.title) != 1:
      logger.warning("Ignoring the title for the evaluation set, as no evaluation set is given")
    if args.eval_files is not None and len(args.title) < 2:
      logger.error("The title for the evaluation set is not specified")

  return args


def _plot_roc(frrs, colors, labels, title, fontsize=18, position=None):
  if position is None: position = 4
  figure = pyplot.figure()
  # plot FMR and 1 - FNMR for each algorithm
  for i in range(len(frrs)):
    pyplot.semilogx([100.0*f for f in frrs[i][0]], [100. - 100.0*f for f in frrs[i][1]], color=colors[i], lw=2, ms=10, mew=1.5, label=labels[i])

  # finalize plot
  pyplot.plot([0.1,0.1],[0,105], "--", color=(0.3,0.3,0.3))
  pyplot.axis([frrs[0][0][0]*100,100,0,105])
  pyplot.xticks((0.01, 0.1, 1, 10, 100), ('0.01', '0.1', '1', '10', '100'))
  pyplot.xlabel('FMR (\%)')
  pyplot.ylabel('1 - FNMR (\%)')
  pyplot.grid(True, color=(0.6,0.6,0.6))
  pyplot.legend(loc=position, prop = {'size':fontsize})
  pyplot.title(title)

  return figure


def _plot_det(dets, colors, labels, title, fontsize=18, position=None):
  if position is None: position = 1
  # open new page for current plot
  figure = pyplot.figure(figsize=(8.2,8))

  # plot the DET curves
  for i in range(len(dets)):
    pyplot.plot(dets[i][0], dets[i][1], color=colors[i], lw=2, ms=10, mew=1.5, label=labels[i])

  # change axes accordingly
  det_list = [0.0002, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9, 0.95]
  ticks = [bob.measure.ppndf(d) for d in det_list]
  labels = [("%.5f" % (d*100)).rstrip('0').rstrip('.') for d in det_list]
  pyplot.xticks(ticks, labels)
  pyplot.yticks(ticks, labels)
  pyplot.axis((ticks[0], ticks[-1], ticks[0], ticks[-1]))

  pyplot.xlabel('FMR (\%)')
  pyplot.ylabel('1 - FNMR (\%)')
  pyplot.legend(loc=position, prop = {'size':fontsize})
  pyplot.title(title)

  return figure

def _plot_cmc(cmcs, colors, labels, title, fontsize=18, position=None):
  if position is None: position = 4
  # open new page for current plot
  figure = pyplot.figure()

  max_x = 0
  # plot the DET curves
  for i in range(len(cmcs)):
    x = bob.measure.plot.cmc(cmcs[i], figure=figure, color=colors[i], lw=2, ms=10, mew=1.5, label=labels[i])
    max_x = max(x, max_x)

  # change axes accordingly
  ticks = [int(t) for t in pyplot.xticks()[0]]
  pyplot.xlabel('Rank')
  pyplot.ylabel('Probability (\%)')
  pyplot.xticks(ticks, [str(t) for t in ticks])
  pyplot.axis([0, max_x, 0, 100])
  pyplot.legend(loc=position, prop = {'size':fontsize})
  pyplot.title(title)

  return figure


def _plot_epc(scores_dev, scores_eval, colors, labels, title, fontsize=18, position=None):
  if position is None: position = 4
  # open new page for current plot
  figure = pyplot.figure()

  # plot the DET curves
  for i in range(len(scores_dev)):
    bob.measure.plot.epc(scores_dev[i][0], scores_dev[i][1], scores_eval[i][0], scores_eval[i][1], 100, color=colors[i], label=labels[i], lw=2)

  # change axes accordingly
  pyplot.xlabel('alpha')
  pyplot.ylabel('HTER (\\%)')
  pyplot.title(title)
  pyplot.grid(True)
  pyplot.legend(loc=position, prop = {'size':fontsize})
  pyplot.title(title)

  return figure


def _plot_distributions(imp_scores, gen_scores, title):
  # Plots genuine and impostor score distributions
  #position = 4
  # open new page for current plot
  figure = pyplot.figure()

  # plot the genuine and impostor distributions
  #pyplot.hist(imp_scores, bins='auto', color='red', normed=True)
  #pyplot.hist(gen_scores, bins='auto', color='green', normed=True)

  add_cnt = 1
  # Impostor score probability distributions
  imp_counts, imp_bin_edges = numpy.histogram(imp_scores, bins=numpy.arange(0.000, 1.005, 0.005))
  # # Normalize it, so that every bins value gives the probability of that bin
  # num_imp_bins = len(numpy.arange(0.00, 1.01, 0.01)) - 1
  # imp_bin_probs = (imp_counts + add_cnt)/(float(len(imp_scores)) + add_cnt*num_imp_bins)
  # print('Number of imp counts initially = %s' % (len(imp_scores)))
  # print('Number of imp counts finally = %s' % (float(len(imp_scores)) + add_cnt*num_imp_bins))
  imp_bin_probs = imp_counts/(float(len(imp_scores)))
  # # Get the mid points of every bin
  imp_bin_middles = (imp_bin_edges[1:]+imp_bin_edges[:-1])/float(2)
  # # Compute the bin-width
  imp_bin_width = imp_bin_edges[1]-imp_bin_edges[0]
  pyplot.bar(imp_bin_middles, imp_bin_probs, width=imp_bin_width, color='red', alpha=0.4, label='Observed Impostor')

  x_imp = numpy.linspace(min(imp_scores) - 0.05, max(imp_scores) + 0.05, 100)
  mean_imp_bins = sum(imp_bin_middles * imp_counts) / sum(imp_counts)
  #print('bin mean = %s' % (mean_imp_bins))
  #print('scores mean = %s' % (numpy.mean(imp_scores)))
  std_dev_imp_bins = math.sqrt(sum(((imp_bin_middles - mean_imp_bins) ** 2) * imp_counts) / sum(imp_counts))
  #print('bin std dev = %s' % (std_dev_imp_bins))
  #print('scores std dev = %s' % (numpy.std(imp_scores)))
  imp_norm_pdf_values = norm.pdf(x_imp, mean_imp_bins, std_dev_imp_bins)
  imp_scale = max(imp_bin_probs) / max(imp_norm_pdf_values)
  imp_scaled_norm_pdf_values = imp_norm_pdf_values * imp_scale
  pyplot.plot(x_imp, imp_scaled_norm_pdf_values, 'r-', lw=1, label='Normal Impostor')

  # Genuine score probability distributions
  gen_counts, gen_bin_edges = numpy.histogram(gen_scores, bins=numpy.arange(0.000, 1.005, 0.005))
  # # Normalize it, so that every bins value gives the probability of that bin
  # num_gen_bins = len(numpy.arange(0.00, 1.01, 0.01)) - 1
  # gen_bin_probs = (gen_counts + add_cnt)/(float(len(gen_scores)) + add_cnt*num_gen_bins)
  # print('Number of gen counts initially = %s' % (len(gen_scores)))
  # print('Number of gen counts finally = %s' % (float(len(gen_scores)) + add_cnt*num_gen_bins))
  gen_bin_probs = gen_counts/(float(len(gen_scores)))
  # # Get the mid points of every bin
  gen_bin_middles = (gen_bin_edges[1:]+gen_bin_edges[:-1])/float(2)
  # # Compute the bin-width
  gen_bin_width = gen_bin_edges[1]-gen_bin_edges[0]
  pyplot.bar(gen_bin_middles, gen_bin_probs, width=gen_bin_width, color='green', alpha=0.4, label='Observed Genuine')

  x_gen = numpy.linspace(min(gen_scores) - 0.05, max(gen_scores) + 0.05, 100)
  mean_gen_bins = sum(gen_bin_middles * gen_counts) / sum(gen_counts)
  #print('bin mean = %s' % (mean_gen_bins))
  #print('scores mean = %s' % (numpy.mean(gen_scores)))
  std_dev_gen_bins = math.sqrt(sum(((gen_bin_middles - mean_gen_bins) ** 2) * gen_counts) / sum(gen_counts))
  #print('bin std dev = %s' % (std_dev_gen_bins))
  #print('scores std dev = %s' % (numpy.std(gen_scores)))
  gen_norm_pdf_values = norm.pdf(x_gen, mean_gen_bins, std_dev_gen_bins)
  gen_scale = max(gen_bin_probs) / max(gen_norm_pdf_values)
  gen_scaled_norm_pdf_values = gen_norm_pdf_values * gen_scale
  pyplot.plot(x_gen, gen_scaled_norm_pdf_values, 'g-', lw=1, label='Normal Genuine')

  # # Calculate relative entropy (Kellback-Leibler divergence):
  # # Correct for zero probabilities in the two distributions
  # P_temp = []
  # Q_temp = []
  # for ind in range(len(imp_bin_probs)):
  #   #print(ind)
  #   if (imp_bin_probs[ind] == 0) and (gen_bin_probs[ind] == 0):
  #     #print('both zero')
  #     continue
  #   elif ((imp_bin_probs[ind] == 0) and (gen_bin_probs[ind] != 0)) or ((imp_bin_probs[ind] != 0) and (gen_bin_probs[ind] == 0)):
  #     #print('one zero and the other non-zero')
  #     Q_temp.append(imp_bin_probs[ind])
  #     P_temp.append(gen_bin_probs[ind])
  # #print('Q_temp: %s' % (Q_temp))
  # #print('P_temp: %s' % (P_temp))
  # epsilon = 10**(-100)
  # num_zero_imp_probs = Q_temp.count(0)
  # num_nonzero_imp_probs = len(Q_temp) - num_zero_imp_probs
  # Q = numpy.array(Q_temp)
  # Q[Q != 0] = Q[Q != 0] - epsilon*num_zero_imp_probs/num_nonzero_imp_probs
  # Q[Q == 0] = epsilon
  # num_zero_gen_probs = P_temp.count(0)
  # num_nonzero_gen_probs = len(P_temp) - num_zero_gen_probs
  # P = numpy.array(P_temp)
  # P[P != 0] = P[P != 0] - epsilon*num_zero_gen_probs/num_nonzero_gen_probs
  # P[P == 0] = epsilon
  # #print('Q: %s' % (Q))
  # #print('P: %s' % (P))
  # # Calculate the relative entropy (KL divergence)
  # #rel_entr = scipy.stats.entropy(P, Q)
  # rel_entr = scipy.stats.entropy(gen_norm_pdf_values, imp_norm_pdf_values)
  # print('Relative entropy = %s' % (rel_entr))

  # # t_imp = scipy.stats.norm(mean_imp_bins, std_dev_imp_bins)
  # # t_gen = scipy.stats.norm(mean_gen_bins, std_dev_gen_bins)

  # # # domain to evaluate PDF on
  # # x = numpy.linspace(0.000, 1.005, 200)

  # # pyplot.plot(x, t_gen.pdf(x) * gen_scale, '-k')
  # # pyplot.plot(x, t_imp.pdf(x) * imp_scale, '-b')

  # # new_rel_entr = scipy.stats.entropy(t_gen.pdf(x), t_imp.pdf(x))
  # # print('new_rel_entr = %s' % (new_rel_entr))

  # # change axes accordingly
  pyplot.xlabel('Match Score')
  pyplot.ylabel('Probability')
  pyplot.title(title)
  pyplot.legend()

  # #print(scipy.stats.mstats.normaltest(imp_scores))
  # #print(scipy.stats.mstats.normaltest(gen_scaled_norm_pdf_values))

  # Calculate relative entropy based on the Nearest-Neighbour estimator for KL-divergence
  import relative_entropy as rl
  rel_entr = rl.calc_NN_estimator(gen_scores, imp_scores)
  print('Relative entropy based on NN-estimator = %s' % (rel_entr))

  return figure



def main(command_line_parameters=None):
  """Reads score files, computes error measures and plots curves."""

  args = command_line_arguments(command_line_parameters)

  # get some colors for plotting
  cmap = pyplot.cm.get_cmap(name='hsv')
  colors = [cmap(i) for i in numpy.linspace(0, 1.0, len(args.dev_files)+1)]

  if args.criterion or args.roc or args.det or args.epc or args.cllr or args.mindcf or args.GIdistr:
    score_parser = {'4column' : bob.measure.load.split_four_column, '5column' : bob.measure.load.split_five_column}[args.parser]

    # First, read the score files
    logger.info("Loading %d score files of the development set", len(args.dev_files))
    scores_dev = [score_parser(os.path.join(args.directory, f)) for f in args.dev_files]

    if args.eval_files:
      logger.info("Loading %d score files of the evaluation set", len(args.eval_files))
      scores_eval = [score_parser(os.path.join(args.directory, f)) for f in args.eval_files]


    if args.criterion:
      logger.info("Computing %s on the development " % args.criterion + ("and HTER on the evaluation set" if args.eval_files else "set"))
      for i in range(len(scores_dev)):
        # compute threshold on development set
        if args.criterion == 'FAR':
          threshold = bob.measure.far_threshold(scores_dev[i][0], scores_dev[i][1], args.far_value/100.)
        else:
          threshold = {'EER': bob.measure.eer_threshold, 'HTER' : bob.measure.min_hter_threshold} [args.criterion](scores_dev[i][0], scores_dev[i][1])
        # apply threshold to development set
        far, frr = bob.measure.farfrr(scores_dev[i][0], scores_dev[i][1], threshold)
        if args.criterion == 'FAR':
          print("The FRR at FAR=%2.3f%% of the development set of '%s' is %2.3f%% (CAR: %2.3f%%)" % (args.far_value, args.legends[i], frr * 100., 100.*(1-frr)))
        else:
          print("The %s of the development set of '%s' is %2.3f%%" % (args.criterion, args.legends[i], (far + frr) * 50.)) # / 2 * 100%
        if args.eval_files:
          # apply threshold to evaluation set
          far, frr = bob.measure.farfrr(scores_eval[i][0], scores_eval[i][1], threshold)
          if args.criterion == 'FAR':
            print("The FRR of the evaluation set of '%s' is %2.3f%% (CAR: %2.3f%%)" % (args.legends[i], frr * 100., 100.*(1-frr))) # / 2 * 100%
          else:
            print("The HTER of the evaluation set of '%s' is %2.3f%%" % (args.legends[i], (far + frr) * 50.)) # / 2 * 100%


    if args.mindcf:
      logger.info("Computing minDCF on the development " + ("and on the evaluation set" if args.eval_files else "set"))
      for i in range(len(scores_dev)):
        # compute threshold on development set
        threshold = bob.measure.min_weighted_error_rate_threshold(scores_dev[i][0], scores_dev[i][1], args.cost)
        # apply threshold to development set
        far, frr = bob.measure.farfrr(scores_dev[i][0], scores_dev[i][1], threshold)
        print("The minDCF of the development set of '%s' is %2.3f%%" % (args.legends[i], (args.cost * far + (1-args.cost) * frr) * 100. ))
        if args.eval_files:
          # compute threshold on evaluation set
          threshold = bob.measure.min_weighted_error_rate_threshold(scores_eval[i][0], scores_eval[i][1], args.cost)
          # apply threshold to evaluation set
          far, frr = bob.measure.farfrr(scores_eval[i][0], scores_eval[i][1], threshold)
          print("The minDCF of the evaluation set of '%s' is %2.3f%%" % (args.legends[i], (args.cost * far + (1-args.cost) * frr) * 100. ))


    if args.cllr:
      logger.info("Computing Cllr and minCllr on the development " + ("and on the evaluation set" if args.eval_files else "set"))
      for i in range(len(scores_dev)):
        cllr = bob.measure.calibration.cllr(scores_dev[i][0], scores_dev[i][1])
        min_cllr = bob.measure.calibration.min_cllr(scores_dev[i][0], scores_dev[i][1])
        print("Calibration performance on development set of '%s' is Cllr %1.5f and minCllr %1.5f " % (args.legends[i], cllr, min_cllr))
        if args.eval_files:
          cllr = bob.measure.calibration.cllr(scores_eval[i][0], scores_eval[i][1])
          min_cllr = bob.measure.calibration.min_cllr(scores_eval[i][0], scores_eval[i][1])
          print("Calibration performance on evaluation set of '%s' is Cllr %1.5f and minCllr %1.5f" % (args.legends[i], cllr, min_cllr))


    if args.roc:
      logger.info("Computing CAR curves on the development " + ("and on the evaluation set" if args.eval_files else "set"))
      fars = [math.pow(10., i * 0.25) for i in range(-16,0)] + [1.]
      frrs_dev = [bob.measure.roc_for_far(scores[0], scores[1], fars) for scores in scores_dev]
      if args.eval_files:
        frrs_eval = [bob.measure.roc_for_far(scores[0], scores[1], fars) for scores in scores_eval]

      logger.info("Plotting ROC curves to file '%s'", args.roc)
      try:
        # create a multi-page PDF for the ROC curve
        pdf = PdfPages(args.roc)
        # create a separate figure for dev and eval
        pdf.savefig(_plot_roc(frrs_dev, colors, args.legends, args.title[0] if args.title is not None else "ROC curve for development set", args.legend_font_size, args.legend_position))
        del frrs_dev
        if args.eval_files:
          pdf.savefig(_plot_roc(frrs_eval, colors, args.legends, args.title[1] if args.title is not None else "ROC curve for evaluation set", args.legend_font_size, args.legend_position))
          del frrs_eval
        pdf.close()
      except RuntimeError as e:
        raise RuntimeError("During plotting of ROC curves, the following exception occured:\n%s\nUsually this happens when the label contains characters that LaTeX cannot parse." % e)

    if args.det:
      logger.info("Computing DET curves on the development " + ("and on the evaluation set" if args.eval_files else "set"))
      dets_dev = [bob.measure.det(scores[0], scores[1], 1000) for scores in scores_dev]
      if args.eval_files:
        dets_eval = [bob.measure.det(scores[0], scores[1], 1000) for scores in scores_eval]

      logger.info("Plotting DET curves to file '%s'", args.det)
      try:
        # create a multi-page PDF for the ROC curve
        pdf = PdfPages(args.det)
        # create a separate figure for dev and eval
        pdf.savefig(_plot_det(dets_dev, colors, args.legends, args.title[0] if args.title is not None else "DET plot for development set", args.legend_font_size, args.legend_position))
        del dets_dev
        if args.eval_files:
          pdf.savefig(_plot_det(dets_eval, colors, args.legends, args.title[1] if args.title is not None else "DET plot for evaluation set", args.legend_font_size, args.legend_position))
          del dets_eval
        pdf.close()
      except RuntimeError as e:
        raise RuntimeError("During plotting of ROC curves, the following exception occured:\n%s\nUsually this happens when the label contains characters that LaTeX cannot parse." % e)


    if args.epc:
      logger.info("Plotting EPC curves to file '%s'", args.epc)

      if not args.eval_files:
        raise ValueError("To plot the EPC curve the evaluation scores are necessary. Please, set it with the --eval-files option.")

      try:
        # create a multi-page PDF for the ROC curve
        pdf = PdfPages(args.epc)
        pdf.savefig(_plot_epc(scores_dev, scores_eval, colors, args.legends, args.title if args.title is not None else "EPC Curves" , args.legend_font_size, args.legend_position))
        pdf.close()
      except RuntimeError as e:
        raise RuntimeError("During plotting of EPC curves, the following exception occured:\n%s\nUsually this happens when the label contains characters that LaTeX cannot parse." % e)


    if args.GIdistr:
      logger.info("Computing genuine and impostor score distributions on the development " + ("and on the evaluation set" if args.eval_files else "set"))
      imp_scores = scores_dev[0][0]
      gen_scores = scores_dev[0][1]
      logger.info("Plotting genuine and impostor distributions to file '%s'", args.GIdistr)
      # create a multi-page PDF for the distributions
      pdf = PdfPages(args.GIdistr)
      # create a separate figure for dev and eval
      pdf.savefig(_plot_distributions(imp_scores, gen_scores, args.title[0]))
      pdf.close()


  if args.cmc or args.rr:
    logger.info("Loading CMC data on the development " + ("and on the evaluation set" if args.eval_files else "set"))
    cmc_parser = {'4column' : bob.measure.load.cmc_four_column, '5column' : bob.measure.load.cmc_five_column}[args.parser]
    cmcs_dev = [cmc_parser(os.path.join(args.directory, f)) for f in args.dev_files]
    if args.eval_files:
      cmcs_eval = [cmc_parser(os.path.join(args.directory, f)) for f in args.eval_files]

    if args.cmc:
      logger.info("Plotting CMC curves to file '%s'", args.cmc)
      try:
        # create a multi-page PDF for the ROC curve
        pdf = PdfPages(args.cmc)
        # create a separate figure for dev and eval
        pdf.savefig(_plot_cmc(cmcs_dev, colors, args.legends, args.title[0] if args.title is not None else "CMC curve for development set", args.legend_font_size, args.legend_position))
        if args.eval_files:
          pdf.savefig(_plot_cmc(cmcs_eval, colors, args.legends, args.title[1] if args.title is not None else "CMC curve for evaluation set", args.legend_font_size, args.legend_position))
        pdf.close()
      except RuntimeError as e:
        raise RuntimeError("During plotting of ROC curves, the following exception occured:\n%s\nUsually this happens when the label contains characters that LaTeX cannot parse." % e)

    if args.rr:
      logger.info("Computing recognition rate on the development " + ("and on the evaluation set" if args.eval_files else "set"))
      for i in range(len(cmcs_dev)):
        rr = bob.measure.recognition_rate(cmcs_dev[i], args.thresholds[i])
        print("The Recognition Rate of the development set of '%s' is %2.3f%%" % (args.legends[i], rr * 100.))
        if args.eval_files:
          rr = bob.measure.recognition_rate(cmcs_eval[i], args.thresholds[i])
          print("The Recognition Rate of the development set of '%s' is %2.3f%%" % (args.legends[i], rr * 100.))
