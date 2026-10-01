# Home Credit sin variables de historial crediticio previo

- Origen: `home_credit_train_filtered.parquet`; se conserva intacto.
- Casos: 1,526,659; `case_id` únicos: 1,526,659.
- Predictores candidatos iniciales: 182.
- Se filtraron variables, no filas: la tabla no identifica una cohorte verificada de solicitantes primerizos.
- Variables retiradas en total: 110; las razones se superponen.
- Variables marcadas por historial previo: 103.
- Variables excluidas por disponibilidad temporal no confirmada: 3.
- Variables de fecha con algún valor mayor que `date_decision`, apartadas preventivamente: 13.
- Predictores conservados: 72; columnas finales, incluidos metadatos y target: 77.
- Dataset generado: `C:\Users\Usuario\Documents\ChatGPT\ML Project 1\data\processed\home_credit_train_no_history_features.parquet`.
- `target`, `case_id` y fechas/calendario se conservan como target o metadatos; no son predictores.

## Revisión temporal

Kaggle define `D` como una transformación de fecha; no significa que la variable use información futura. El anfitrión indica que la predicción ocurre en `date_decision` y que los datos están disponibles en ese momento ([convención de sufijos](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data), [aclaración del anfitrión](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/472866)).
Algunas columnas contienen valores de fecha mayores que `date_decision`. Eso por sí solo no demuestra cuándo se conoció el dato: puede ser una fecha de evento futura conocida al solicitar el crédito. Se apartaron preventivamente para revisarlas por variable; no se etiquetan automáticamente como leakage.
También se apartaron campos de desembolso, cuota del mes siguiente y primera fecha de vencimiento porque sus descripciones no confirman que estén disponibles antes de decidir.
`target` se conserva como etiqueta: representa el resultado del crédito observado después de la decisión y nunca entra entre los predictores.

## Variables con valores de fecha mayores que `date_decision`, apartadas para revisar

- `validfrom_1069D`: 54,689 filas con valor de fecha mayor.
- `lastdelinqdate_224D`: 29,245 filas con valor de fecha mayor.
- `dtlastpmtallstes_4499206D`: 126,627 filas con valor de fecha mayor.
- `empl_employedfrom_271D`: 316 filas con valor de fecha mayor.
- `datelastunpaid_3546854D`: 28,221 filas con valor de fecha mayor.
- `firstclxcampaign_1125D`: 37,006 filas con valor de fecha mayor.
- `datefirstoffer_1144D`: 1,942 filas con valor de fecha mayor.
- `maxdpdinstldate_3546855D`: 12,710 filas con valor de fecha mayor.
- `lastrejectdate_50D`: 38,799 filas con valor de fecha mayor.
- `firstdatedue_489D`: 6,912 filas con valor de fecha mayor.
- `lastactivateddate_801D`: 15,219 filas con valor de fecha mayor.
- `lastapprdate_640D`: 34,241 filas con valor de fecha mayor.
- `lastapplicationdate_877D`: 114,236 filas con valor de fecha mayor.

## Otras variables retiradas

- `datelastinstal40dpd_247D` — Historial de crédito o pagos previos. Date of last instalment that was more than 40 days past due (DPD).
- `avglnamtstart24m_4525187A` — Historial de crédito o pagos previos. Average loan amount in the last 24 months.
- `cardtype_51L` — Historial de crédito o pagos previos. Type of credit card.
- `totinstallast1m_4525188A` — Historial de crédito o pagos previos. Total amount of monthly instalments paid in the previous month.
- `maxpmtlast3m_4525190A` — Historial de crédito o pagos previos. Maximum payment made by the client in the last 3 months.
- `maxlnamtstart6m_4525199A` — Historial de crédito o pagos previos. Maximum loan amount started in the last 6 months.
- `avgpmtlast12m_4525200A` — Historial de crédito o pagos previos. Average of payments made by the client in the last 12 months.
- `avgdbdtollast24m_4525197P` — Historial de crédito o pagos previos. Average days of payment before due date within the last 24 months (with tolerance).
- `mindbdtollast24m_4525191P` — Historial de crédito o pagos previos. Minimum days before due in last 24 months.
- `maxdbddpdlast1m_3658939P` — Historial de crédito o pagos previos. Maximum number of days past due in the last month. A negative value indicates the number of days before the due date.
- `avgdbddpdlast3m_4187120P` — Historial de crédito o pagos previos. Average days past or before due of payment during the last 3 months.
- `maxdbddpdtollast6m_4187119P` — Historial de crédito o pagos previos. Maximum number of days past due in last 6 months. This predictor takes the value as a negative number when it represents days before due date.
- `numinstmatpaidtearly2d_4499204L` — Historial de crédito o pagos previos. Number of instalments that have been paid more than 2 days before their due date.
- `numinstpaid_4499208L` — Historial de crédito o pagos previos. Number of paid instalments.
- `numinstpaidearly5dobd_4499205L` — Historial de crédito o pagos previos. Number of installments paid more than 5 days prior to the due date.
- `avgoutstandbalancel6m_4187114A` — Historial de crédito o pagos previos. Average outstanding balance of applicant for the last 6 months.
- `numinstpaidearly3dest_4493216L` — Historial de crédito o pagos previos. Number of instalments that have been paid more than 3 days in advance of the due date.
- `numinstpaidearly5dest_4493211L` — Historial de crédito o pagos previos. Number of instalments that were paid more than 5 days before the due date.
- `numinstpaidearlyest_4493214L` — Historial de crédito o pagos previos. Number of instalments paid before the due date.
- `numinstregularpaidest_4493210L` — Historial de crédito o pagos previos. Number of fully paid regular installments on clients' previous contracts.
- `numinsttopaygrest_4493213L` — Historial de crédito o pagos previos. Number of unpaid instalments.
- `numinstunpaidmaxest_4493212L` — Historial de crédito o pagos previos. Maximum number of unpaid instalments.
- `sumoutstandtotalest_4493215A` — Historial de crédito o pagos previos. Sum of total outstanding amount.
- `maxdpdinstlnum_3546846P` — Historial de crédito o pagos previos. Instalment number of which client was most days past due.
- `lastrejectcredamount_222A` — Historial de crédito o pagos previos. Credit amount on last rejected application.
- `maxoutstandbalancel12m_4187113A` — Historial de crédito o pagos previos. Maximum outstanding balance in the last 12 months.
- `avgmaxdpdlast9m_3716943P` — Historial de crédito o pagos previos. Average Days Past Due (DPD) of the client in last 9 months.
- `maxdbddpdtollast12m_3658940P` — Historial de crédito o pagos previos. Maximum number of days past due in last 12 months. A negative value implies days before due date.
- `numinstpaidlastcontr_4325080L` — Historial de crédito o pagos previos. Number of paid installments from the client's last contract.
- `avginstallast24m_3658937A` — Historial de crédito o pagos previos. Average instalments paid by the client over the past 24 months.
- `maxinstallast24m_3658928A` — Historial de crédito o pagos previos. Maximum instalment in the last 24 months
- `avgdbddpdlast24m_3658932P` — Historial de crédito o pagos previos. Average days past or before due of payment during the last 24 months.
- `mindbddpdlast24m_3658935P` — Historial de crédito o pagos previos. Minimum days past due (or days before due) in last 24 months.
- `amtinstpaidbefduel24m_4187115A` — Historial de crédito o pagos previos. Number of instalments paid before due date in the last 24 months.
- `numinstlswithdpd5_4187116L` — Historial de crédito o pagos previos. Number of instalments that were overdue by at least 5 days.
- `avgdpdtolclosure24_3658938P` — Historial de crédito o pagos previos. Average DPD (days past due) with tolerance within the past 24 months from the maximum closure date, assuming that the contract is finished. If the contract is ongoing, the calculation is based on the current date.
- `pctinstlsallpaidlat10d_839L` — Historial de crédito o pagos previos. Percentage of installments that were paid 10 or more days after the due date.
- `cntpmts24_3658933L` — Historial de crédito o pagos previos. Number of months with any incoming payment in last 24 months.
- `pctinstlsallpaidlate6d_3546844L` — Historial de crédito o pagos previos. Percentage of installments that were paid 6 or more days past their due date.
- `pctinstlsallpaidlate4d_3546849L` — Historial de crédito o pagos previos. Percentage of installments that were paid 4 or more days past their due date.
- `pctinstlsallpaidearl3d_427L` — Historial de crédito o pagos previos. Percentage of installments paid at least 3 days prior to the due date.
- `pctinstlsallpaidlate1d_3546856L` — Historial de crédito o pagos previos. Percentage of installments that are paid 1 or more days after the due date.
- `numinstlswithdpd10_728L` — Historial de crédito o pagos previos. Number of instalments that were overdue for 10 or more days.
- `numinstlswithoutdpd_562L` — Historial de crédito o pagos previos. Number of instalments that were not past due date.
- `numinstregularpaid_973L` — Historial de crédito o pagos previos. Number of fully paid regular installments in the client's previous contracts.
- `cntincpaycont9m_3716944L` — Historial de crédito o pagos previos. Number of incoming payments in the past 9 months.
- `numincomingpmts_3546848L` — Historial de crédito o pagos previos. Number of incoming payments.
- `daysoverduetolerancedd_3976961L` — Historial de crédito o pagos previos. Number of days that past after the due date (with tolerance).
- `monthsannuity_845L` — Historial de crédito o pagos previos. Monthly annuity amount for the applicant.
- `numinstpaidearly5d_1087L` — Historial de crédito o pagos previos. Number of instalments paid more than 5 days prior to the due date.
- `numinstpaidearly_338L` — Historial de crédito o pagos previos. Number of installments paid prior to the due date.
- `numinstpaidlate1d_3546852L` — Historial de crédito o pagos previos. Number of instalments paid more than 1 day past their due date.
- `numinsttopaygr_769L` — Historial de crédito o pagos previos. Number of unpaid instalments.
- `numinstunpaidmax_3546851L` — Historial de crédito o pagos previos. Maximum number of unpaid instalments.
- `numinstpaidearly3d_3546850L` — Historial de crédito o pagos previos. Number of instalments paid more than three days before the due date.
- `numinstlallpaidearly3d_817L` — Historial de crédito o pagos previos. Number of instalments paid at least 3 days prior to their due date.
- `numinstlsallpaid_934L` — Historial de crédito o pagos previos. Number of paid instalments.
- `sumoutstandtotal_3546847A` — Historial de crédito o pagos previos. Sum of total outstanding amount.
- `lastapprcredamount_781A` — Historial de crédito o pagos previos. Credit amount from the client's last application.
- `actualdpdtolerance_344P` — Historial de crédito o pagos previos. DPD of client with tolerance.
- `commnoinclast6m_3546845L` — Historial de crédito o pagos previos. Number of communications indicating low income in the last six months.
- `maxdpdfrom6mto36m_3546853P` — Historial de crédito o pagos previos. Maximum Days Past Due (DPD) in the period ranging from 6 to 36 months.
- `mastercontrelectronic_519L` — Historial de crédito o pagos previos. Flag indicating the existence of the master contract for the client.
- `mastercontrexist_109L` — Historial de crédito o pagos previos. Flag indicating whether or not the applicant has an existing master contract.
- `maxannuity_159A` — Historial de crédito o pagos previos. Maximum annuity previously obtained by client.
- `maxdebt4_972A` — Historial de crédito o pagos previos. Maximal principal debt of the client in the history older than 4 months.
- `maxdpdlast12m_727P` — Historial de crédito o pagos previos. Maximum days past due in the past 12 months.
- `maxdpdlast24m_143P` — Historial de crédito o pagos previos. Maximal days past due in the last 24 months.
- `maxdpdlast3m_392P` — Historial de crédito o pagos previos. Maximum number of days past due in last 3 months.
- `maxdpdlast6m_474P` — Historial de crédito o pagos previos. Maximum days past due in the last 6 months.
- `maxdpdlast9m_1059P` — Historial de crédito o pagos previos. Maximum days past due in last 9 months.
- `maxdpdtolerance_374P` — Historial de crédito o pagos previos. Maximum number of days past due (with tolerance).
- `lastst_736L` — Historial de crédito o pagos previos. Status of the client's previous credit application.
- `opencred_647L` — Historial de crédito o pagos previos. Number of active loans from the previous application.
- `interestrate_311L` — Historial de crédito o pagos previos. The interest rate of the active credit contract.
- `pmtnum_254L` — Historial de crédito o pagos previos. Total number of loan payments made by the client.
- `annuitynextmonth_57A` — Disponibilidad temporal no confirmada. Next month's amount of annuity.
- `currdebt_22A` — Historial de crédito o pagos previos. Current debt amount of the client.
- `currdebtcredtyperange_828A` — Historial de crédito o pagos previos. Current amount of debt of the applicant.
- `disbursedcredamount_1113A` — Disponibilidad temporal no confirmada. Disbursed credit amount after consolidation.
- `lastapprcommoditycat_1041M` — Historial de crédito o pagos previos. Commodity category of the last loan applications made by the applicant.
- `lastapprcommoditytypec_5251766M` — Historial de crédito o pagos previos. Commodity type of the last application.
- `lastcancelreason_561M` — Historial de crédito o pagos previos. Cancellation reason of the last application.
- `lastrejectcommoditycat_161M` — Historial de crédito o pagos previos. Category of commodity in the applicant's last rejected application.
- `lastrejectcommodtypec_5251769M` — Historial de crédito o pagos previos. Commodity type of the last rejected application.
- `lastrejectreason_759M` — Historial de crédito o pagos previos. Reason for rejection on the most recent rejected application.
- `lastrejectreasonclient_4145040M` — Historial de crédito o pagos previos. Reason for the client's last loan rejection.
- `numactivecreds_622L` — Historial de crédito o pagos previos. Number of active credits.
- `numactivecredschannel_414L` — Historial de crédito o pagos previos. Number of active credits.
- `numactiverelcontr_750L` — Historial de crédito o pagos previos. Number of active revolving credits.
- `numcontrs3months_479L` — Historial de crédito o pagos previos. Number of contracts in last 3 months.
- `numnotactivated_1143L` — Historial de crédito o pagos previos. Number of non-activated credits.
- `numpmtchanneldd_318L` — Historial de crédito o pagos previos. Number of previous loan contracts for the applicant that had direct debit as payment channel.
- `numrejects9m_859L` — Historial de crédito o pagos previos. Number of credit applications that were rejected in the last 9 months.
- `previouscontdistrict_112M` — Historial de crédito o pagos previos. Contact district of the client's previous approved application.
- `totaldebt_9A` — Historial de crédito o pagos previos. Total amount of debt.
- `totalsettled_863A` — Historial de crédito o pagos previos. Sum of all payments made by the client.

Se conservaron las tasas agregadas del punto de venta (`pos*`) porque describen el canal y no los pagos previos de la persona. Los conteos de coincidencias de teléfonos tampoco representan su historial crediticio individual.

Auditoría completa de las 182 variables: `C:\Users\Usuario\Documents\ChatGPT\ML Project 1\reports\metrics\home_credit_first_credit_feature_audit.csv`.
No se entrenó un modelo. Las variables sensibles, incluida `sex_738L`, aún requieren revisión antes de definir los predictores finales.
