//+------------------------------------------------------------------+
//| EURUSD_MA_Ensemble.mq5                                           |
//| V9 M15+M30+H1 only + cooldown v1.07                              |
//+------------------------------------------------------------------+
#property copyright "MA Ensemble EA"
#property version   "1.07"
#property strict

#include <Trade\Trade.mqh>

input int    InpMagicBase = 1234560;
input double InpLot       = 0.01;
input int    InpCooldown  = 3;

CTrade trade;

#define NC 5
#define NTF 3

int M15f[]={26,25,24,26,26};
int M15s[]={44,44,46,45,42};
double M15e[]={0.483,0.483,0.418,0.358,0.599};
double M15sl[]={0.50,0.50,0.50,0.50,0.50};
double M15tp[]={3.51,3.39,3.40,3.45,3.32};

int M30f[]={49,49,49,48,48};
int M30s[]={60,60,60,59,59};
double M30e[]={0.957,0.922,0.919,0.903,0.949};
double M30sl[]={0.56,0.56,0.57,0.57,0.57};
double M30tp[]={3.01,3.01,3.06,3.01,3.00};

int H1f[]={69,67,69,69,68};
int H1s[]={80,80,80,80,80};
double H1e[]={1.118,1.175,1.164,1.162,1.021};
double H1sl[]={0.67,0.56,0.50,0.53,0.59};
double H1tp[]={0.69,3.00,2.93,2.94,0.60};

int hMA[NTF*NC][2];
int hATR[NTF];

datetime lastSLBar[NTF*NC];

//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetDeviationInPoints(10);
   ArrayInitialize(lastSLBar, 0);

   ENUM_TIMEFRAMES tfs[NTF];
   tfs[0]=PERIOD_M15; tfs[1]=PERIOD_M30; tfs[2]=PERIOD_H1;

   for(int t=0;t<NTF;t++)
   {
      hATR[t] = iATR(_Symbol, tfs[t], 14);
      if(hATR[t]==INVALID_HANDLE){Print("ATR fail tf=",t); return INIT_FAILED;}
   }

   int idx;
   for(int c=0;c<NC;c++)
   {
      idx=0*NC+c; hMA[idx][0]=iMA(_Symbol,PERIOD_M15,M15f[c],0,MODE_SMA,PRICE_CLOSE);
                   hMA[idx][1]=iMA(_Symbol,PERIOD_M15,M15s[c],0,MODE_SMA,PRICE_CLOSE);
      idx=1*NC+c; hMA[idx][0]=iMA(_Symbol,PERIOD_M30,M30f[c],0,MODE_SMA,PRICE_CLOSE);
                   hMA[idx][1]=iMA(_Symbol,PERIOD_M30,M30s[c],0,MODE_SMA,PRICE_CLOSE);
      idx=2*NC+c; hMA[idx][0]=iMA(_Symbol,PERIOD_H1,H1f[c],0,MODE_SMA,PRICE_CLOSE);
                   hMA[idx][1]=iMA(_Symbol,PERIOD_H1,H1s[c],0,MODE_SMA,PRICE_CLOSE);
   }

   for(int i=0;i<NTF*NC;i++)
   {
      if(hMA[i][0]==INVALID_HANDLE||hMA[i][1]==INVALID_HANDLE)
      {Print("MA fail idx=",i); return INIT_FAILED;}
   }

   Print("MA Ensemble v1.07 - M15+M30+H1, 15 champions, cooldown=",InpCooldown);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   for(int t=0;t<NTF;t++) IndicatorRelease(hATR[t]);
   for(int i=0;i<NTF*NC;i++){IndicatorRelease(hMA[i][0]);IndicatorRelease(hMA[i][1]);}
}

//+------------------------------------------------------------------+
double GI(int h,int s){double b[];if(CopyBuffer(h,0,s,1,b)<1)return 0;return b[0];}

//+------------------------------------------------------------------+
int PosType(int magic)
{
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong t=PositionGetTicket(i);
      if(t>0 && PositionGetString(POSITION_SYMBOL)==_Symbol
         && PositionGetInteger(POSITION_MAGIC)==magic)
         return (int)PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY?1:-1;
   }
   return 0;
}

//+------------------------------------------------------------------+
ulong FindPos(int magic)
{
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong t=PositionGetTicket(i);
      if(t>0 && PositionGetString(POSITION_SYMBOL)==_Symbol
         && PositionGetInteger(POSITION_MAGIC)==magic)
         return t;
   }
   return 0;
}

//+------------------------------------------------------------------+
void GetParams(int tfIdx, int &fa[], int &sa[], double &ea[], double &sla[], double &tpa[])
{
   if(tfIdx==0){ArrayCopy(fa,M15f);ArrayCopy(sa,M15s);ArrayCopy(ea,M15e);ArrayCopy(sla,M15sl);ArrayCopy(tpa,M15tp);}
   if(tfIdx==1){ArrayCopy(fa,M30f);ArrayCopy(sa,M30s);ArrayCopy(ea,M30e);ArrayCopy(sla,M30sl);ArrayCopy(tpa,M30tp);}
   if(tfIdx==2){ArrayCopy(fa,H1f);ArrayCopy(sa,H1s);ArrayCopy(ea,H1e);ArrayCopy(sla,H1sl);ArrayCopy(tpa,H1tp);}
}

//+------------------------------------------------------------------+
void OnTick()
{
   static datetime lastBar=0;
   datetime curBar=iTime(_Symbol,PERIOD_M5,0);
   if(curBar==lastBar) return;
   lastBar=curBar;

   int fa[NC],sa[NC];
   double ea[NC],sla[NC],tpa[NC];

   for(int t=0;t<NTF;t++)
   {
      GetParams(t,fa,sa,ea,sla,tpa);

      double atr=GI(hATR[t],1);
      if(atr<=0) continue;

      int base=t*NC;

      for(int c=0;c<NC;c++)
      {
         int magic=InpMagicBase + t*10 + c;
         int idx=base+c;

         double maf=GI(hMA[idx][0],1);
         double mas=GI(hMA[idx][1],1);
         if(maf==0||mas==0) continue;

         double diff=(maf-mas)/atr;
         int pt=PosType(magic);

         if(pt==0)
         {
            bool cooled = (lastSLBar[idx]>0 && (curBar-lastSLBar[idx])/300 < InpCooldown);

            if(diff>ea[c] && !cooled)
            {
               double price=SymbolInfoDouble(_Symbol,SYMBOL_ASK);
               int dig=(int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
               double slP=NormalizeDouble(price-atr*sla[c],dig);
               double tpP=NormalizeDouble(price+atr*tpa[c],dig);
               trade.SetExpertMagicNumber(magic);
               trade.Buy(InpLot,_Symbol,price,slP,tpP,StringFormat("V9_%d_%d",t,c));
            }
            else if(diff<-ea[c] && !cooled)
            {
               double price=SymbolInfoDouble(_Symbol,SYMBOL_BID);
               int dig=(int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
               double slP=NormalizeDouble(price+atr*sla[c],dig);
               double tpP=NormalizeDouble(price-atr*tpa[c],dig);
               trade.SetExpertMagicNumber(magic);
               trade.Sell(InpLot,_Symbol,price,slP,tpP,StringFormat("V9_%d_%d",t,c));
            }
         }
         else
         {
            if(pt==1 && diff<-ea[c]*0.3)
            {
               ulong ticket=FindPos(magic);
               if(ticket>0){trade.SetExpertMagicNumber(magic);trade.PositionClose(ticket);}
            }
            else if(pt==-1 && diff>ea[c]*0.3)
            {
               ulong ticket=FindPos(magic);
               if(ticket>0){trade.SetExpertMagicNumber(magic);trade.PositionClose(ticket);}
            }
         }
      }
   }

   // Track SL hits
   static datetime lastDealTime=0;
   HistorySelect(lastDealTime, curBar + 86400);
   int totalDeals = HistoryDealsTotal();
   for(int i=0; i<totalDeals; i++)
   {
      ulong dealTicket = HistoryDealGetTicket(i);
      if(dealTicket==0) continue;
      if(HistoryDealGetString(dealTicket, DEAL_SYMBOL) != _Symbol) continue;
      long dealMagic = HistoryDealGetInteger(dealTicket, DEAL_MAGIC);
      if(dealMagic < InpMagicBase || dealMagic >= InpMagicBase + NTF*NC) continue;
      if(HistoryDealGetInteger(dealTicket, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;
      if(HistoryDealGetInteger(dealTicket, DEAL_REASON) == DEAL_REASON_SL)
      {
         int tfIdx = ((int)dealMagic - InpMagicBase) / NC;
         int chIdx = ((int)dealMagic - InpMagicBase) % NC;
         int arrIdx = tfIdx * NC + chIdx;
         if(arrIdx >= 0 && arrIdx < NTF*NC)
            lastSLBar[arrIdx] = (datetime)HistoryDealGetInteger(dealTicket, DEAL_TIME);
      }
      lastDealTime = (datetime)HistoryDealGetInteger(dealTicket, DEAL_TIME);
   }
}
//+------------------------------------------------------------------+
